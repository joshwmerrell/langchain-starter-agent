import json
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path

from wallace_planning import (
    ArchiveManager,
    ConversationThreadManager,
    PlanExecutor,
    PlanStep,
    SourceRef,
    create_plan,
    load_plan,
    parse_request,
    planning_context,
    resume_or_create,
)


class PlanningTests(unittest.TestCase):
    def test_create_and_round_trip_plan_artifacts(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            plan = parse_request(root, "Implement the change in main.py", session_id="session_test")
            self.assertTrue((root / ".wallace/plans/sessions/session_test").exists())
            self.assertEqual(load_plan(root, plan.session_id, plan.plan_id).request, plan.request)
            self.assertTrue((root / ".wallace/plans/index.json").exists())
            self.assertIn("main.py", [source.path for source in plan.sources])

    def test_resume_skips_completed_steps(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            calls = []

            def fake_tool(value="ok"):
                calls.append(value)
                return f"done:{value}"

            plan = create_plan(
                root,
                "do two things",
                session_id="session_resume",
                steps=[
                    PlanStep("step-01", "first", tool_name="fake", tool_args={"value": "one"}),
                    PlanStep("step-02", "second", tool_name="fake", tool_args={"value": "two"}),
                ],
            )
            executor = PlanExecutor(root, {"fake": fake_tool})
            executor.execute_all(plan)
            executor.execute_all(load_plan(root, plan.session_id, plan.plan_id))
            self.assertEqual(calls, ["one", "two"])
            self.assertEqual(plan.status, "complete")

    def test_failed_step_is_recorded_and_can_be_retried(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            should_fail = {"value": True}

            def flaky_tool():
                if should_fail["value"]:
                    should_fail["value"] = False
                    raise RuntimeError("temporary failure")
                return "recovered"

            plan = create_plan(
                root,
                "recover",
                steps=[PlanStep("step-01", "retry", tool_name="flaky")],
            )
            executor = PlanExecutor(root, {"flaky": flaky_tool})
            step = executor.execute_next(plan)
            self.assertEqual(step.status, "failed")
            self.assertEqual(plan.status, "failed")
            step.status = "pending"
            plan.status = "active"
            executor.execute_next(plan)
            self.assertEqual(plan.status, "complete")
            self.assertEqual(step.result, "recovered")

    def test_rejected_approval_does_not_run_tool(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            calls = []

            def fake_tool():
                calls.append(True)
                return "should not happen"

            plan = create_plan(root, "approval", steps=[PlanStep("step-01", "write", tool_name="fake")])
            step = PlanExecutor(root, {"fake": fake_tool}).execute_next(
                plan, approval_callback=lambda ignored: False
            )
            self.assertEqual(step.status, "failed")
            self.assertEqual(calls, [])

    def test_cleanup_archives_only_completed_plan_and_keeps_active_plan(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            old = (datetime.now(timezone.utc) - timedelta(days=31)).isoformat()
            complete = create_plan(root, "old complete", session_id="session_cleanup")
            complete.steps[0].status = "complete"
            complete.status = "complete"
            complete.updated_at = old
            complete.steps[0].updated_at = old
            from wallace_planning.generators import save_plan

            save_plan(root, complete)
            active = create_plan(root, "keep active", session_id="session_cleanup")
            active.updated_at = old
            save_plan(root, active)
            result = ArchiveManager(root).cleanup()
            self.assertEqual(result["archived"], 1)
            self.assertTrue((root / ".wallace/plans/archive/session_cleanup").exists())
            self.assertTrue((root / ".wallace/plans/sessions/session_cleanup").exists())
            index = json.loads((root / ".wallace/plans/index.json").read_text())
            records = {item["plan_id"]: item for item in index["plans"]}
            self.assertIsNotNone(records[complete.plan_id]["archived_at"])
            self.assertIsNone(records[active.plan_id]["archived_at"])

    def test_cleanup_purges_old_archives_but_retains_metadata(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            plan = create_plan(root, "old archive")
            plan.steps[0].status = "complete"
            plan.status = "complete"
            plan.updated_at = (datetime.now(timezone.utc) - timedelta(days=31)).isoformat()
            from wallace_planning.generators import save_plan

            save_plan(root, plan)
            manager = ArchiveManager(root)
            manager.cleanup()
            archived_json = next((root / ".wallace/plans/archive").glob("*/**/plan_*.json"))
            data = json.loads(archived_json.read_text())
            data["archived_at"] = (datetime.now(timezone.utc) - timedelta(days=91)).isoformat()
            archived_json.write_text(json.dumps(data))
            result = manager.cleanup()
            self.assertEqual(result["purged"], 1)
            self.assertFalse(archived_json.parent.exists())
            records = json.loads((root / ".wallace/plans/index.json").read_text())["plans"]
            self.assertIsNotNone(next(item for item in records if item["plan_id"] == plan.plan_id)["deleted_at"])

    def test_parser_rejects_private_and_parent_sources(self):
        with tempfile.TemporaryDirectory() as directory:
            plan = parse_request(Path(directory), "Review ../secret.py and .wallace/conversation.json")
            self.assertEqual(plan.sources, [])

    def test_direct_plan_creation_rejects_unsafe_source(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(ValueError):
                create_plan(
                    Path(directory),
                    "unsafe",
                    sources=[SourceRef("../secret.py")],
                )

    def test_each_distinct_request_gets_a_plan_but_retries_resume(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first = resume_or_create(root, "Implement main.py")
            retry = resume_or_create(root, "Implement main.py")
            second = resume_or_create(root, "Review tools.py")
            self.assertEqual(first.plan_id, retry.plan_id)
            self.assertNotEqual(first.plan_id, second.plan_id)
            self.assertLess(len(planning_context(first)), 3000)

    def test_explicit_resume_mid_message_reuses_latest_plan(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            original = parse_request(root, "Implement the pipeline: detect language, translate source, validate output", session_id="work")
            resumed = resume_or_create(
                root,
                "You already have the requirements. Resume the latest active plan and implement the next step.",
                session_id="work",
            )
            self.assertEqual(resumed.plan_id, original.plan_id)
            unrelated = resume_or_create(root, "Add a resume button to the TUI", session_id="work")
            self.assertNotEqual(unrelated.plan_id, original.plan_id)
            negated = resume_or_create(root, "Do not resume the latest active plan; review tools.py", session_id="work")
            self.assertNotEqual(negated.plan_id, unrelated.plan_id)

    def test_feature_list_creates_concrete_steps_without_private_path_in_prompt(self):
        with tempfile.TemporaryDirectory() as directory:
            request = (
                "Implement the Python-to-JavaScript pipeline: detection from extensions and contents, "
                "translation, ESLint warnings, strict type-checking, and structural AST comparison."
            )
            plan = parse_request(Path(directory), request, session_id="work")
            titles = [step.title for step in plan.steps]
            self.assertEqual(titles[1:-2], [
                "Implement detection from extensions and contents",
                "Implement translation",
                "Implement ESLint warnings",
                "Implement strict type-checking",
                "Implement structural AST comparison",
            ])
            context = planning_context(plan)
            self.assertIn("Implement strict type-checking", context)
            self.assertNotIn("Plan artifact:", context)
            self.assertNotIn(".wallace/plans/sessions/", context)

    def test_unedited_legacy_plan_is_upgraded_on_resume(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            request = "Implement the pipeline: detect language, translate source, validate output."
            plan = create_plan(
                root, request, intent="implement", session_id="work", steps=[
                    PlanStep("step-01", "Inspect relevant context", status="complete", result="Read main.py"),
                    PlanStep("step-02", "Implement the requested work"),
                    PlanStep("step-03", "Validate the result"),
                    PlanStep("step-04", "Summarize and hand off"),
                ],
            )
            upgraded = resume_or_create(root, "Please resume the latest active plan", session_id="work")
            self.assertEqual(upgraded.plan_id, plan.plan_id)
            self.assertEqual(len(upgraded.steps), 6)
            self.assertEqual(upgraded.steps[0].result, "Read main.py")
            self.assertEqual(upgraded.steps[1].title, "Implement detect language")
            self.assertEqual(load_plan(root, "work", plan.plan_id).steps[1].title, "Implement detect language")

    def test_one_edit_cannot_complete_multiple_feature_steps(self):
        with tempfile.TemporaryDirectory() as directory:
            plan = parse_request(Path(directory), "Implement the pipeline: detect language, translate source, validate output")
            plan.record_progress([
                {"id": "read", "name": "read_safe_file", "target": "main.py", "outcome": "ok", "ok": True},
                {"id": "edit-1", "name": "replace_in_file", "target": "main.py", "outcome": "ok", "ok": True},
                {"id": "test-1", "name": "run_tests", "target": "tests", "outcome": "ok", "ok": True},
            ], response="Implemented and tested language detection.")
            self.assertEqual([step.status for step in plan.steps[1:-2]], ["complete", "pending", "pending"])
            self.assertEqual(plan.steps[-2].status, "pending")
            plan.record_progress([], response="I am still working.")
            self.assertEqual([step.status for step in plan.steps[1:-2]], ["complete", "pending", "pending"])
            plan.record_progress([
                {"id": "edit-2", "name": "replace_in_file", "target": "tools.py", "outcome": "ok", "ok": True},
                {"id": "test-2", "name": "run_tests", "target": "tests", "outcome": "ok", "ok": True},
            ], response="Implemented and tested translation.")
            self.assertEqual([step.status for step in plan.steps[1:-2]], ["complete", "complete", "pending"])
            plan.record_progress([
                {"id": "edit-3", "name": "replace_in_file", "target": "tools.py", "outcome": "ok", "ok": True},
                {"id": "test-3", "name": "run_tests", "target": "tests", "outcome": "ok", "ok": True},
            ], response="Implemented and tested output validation.")
            self.assertEqual(plan.status, "complete")

    def test_implementation_plan_requires_observed_edit_and_validation(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            plan = parse_request(root, "Implement changes in main.py", session_id="work")
            plan.record_progress([], response="I'll start by reading the code.")
            self.assertEqual(plan.status, "active")
            self.assertEqual([step.status for step in plan.steps], ["pending"] * 4)

            plan.record_progress([{
                "id": "read-1", "name": "read_safe_file", "target": "main.py",
                "outcome": "ok", "ok": True,
            }])
            self.assertEqual([step.status for step in plan.steps[:2]], ["complete", "pending"])

            plan.record_progress([{
                "id": "write-1", "name": "replace_in_file", "target": "main.py",
                "outcome": "ok", "ok": True,
            }], response="The edit is in, but I have not tested it.")
            self.assertEqual(plan.status, "active")
            self.assertEqual(plan.steps[2].status, "pending")

            plan.record_progress([{
                "id": "test-1", "name": "run_tests", "target": "tests",
                "outcome": "Exit code: 0", "ok": True,
            }], response="Partial handoff after malformed tool call.", error="malformed tool call")
            self.assertEqual(plan.status, "active")
            self.assertEqual(plan.steps[1].status, "pending")
            plan.record_progress([], response="Edited and tested main.py.")
            self.assertEqual(plan.status, "complete")
            self.assertIn("replace_in_file main.py", planning_context(plan))
            self.assertEqual(len(plan.tool_history), 3)
            from wallace_planning import save_plan
            save_plan(root, plan)
            restored = load_plan(root, plan.session_id, plan.plan_id)
            self.assertEqual(restored.tool_history, plan.tool_history)
            self.assertIn(
                "## Recent tool calls",
                (root / ".wallace/plans/sessions/work" / f"plan_{plan.plan_id}.md").read_text(),
            )

    def test_error_keeps_plan_active_and_continuation_resumes_it(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            plan = parse_request(root, "Implement main.py", session_id="work")
            plan.record_progress([{
                "id": "read-1", "name": "read_safe_file", "target": "main.py",
                "outcome": "ok", "ok": True,
            }], error="invalid tool call arguments")
            from wallace_planning import save_plan
            save_plan(root, plan)
            resumed = resume_or_create(root, "Continue with the roadmap", session_id="work")
            self.assertEqual(resumed.plan_id, plan.plan_id)
            context = planning_context(resumed)
            self.assertIn("Original request: Implement main.py", context)
            self.assertIn("Last error:", context)

    def test_plan_context_does_not_repeat_unverified_assistant_summary(self):
        with tempfile.TemporaryDirectory() as directory:
            plan = parse_request(Path(directory), "Implement main.py", session_id="work")
            plan.record_progress([], response="Missing step_id; use /workspace to proceed.")
            plan.record_progress([], error="Malformed tool call prevented completion.")
            context = planning_context(plan, "Also falsely claimed tool work completed.")
            self.assertIn("Original request: Implement main.py", context)
            self.assertIn("Last error:", context)
            self.assertNotIn("step_id", context)
            self.assertNotIn("/workspace", context)
            self.assertNotIn("falsely claimed", context)

    def test_failed_command_is_labeled_as_failed_in_plan_context(self):
        with tempfile.TemporaryDirectory() as directory:
            plan = parse_request(Path(directory), "Implement main.py", session_id="work")
            plan.record_progress([{
                "id": "failed-command", "name": "execute_command",
                "target": "cd /workspace && pwd", "outcome": "Exit code: 1", "ok": False,
            }])
            context = planning_context(plan)
            self.assertIn("FAILED: execute_command cd /workspace && pwd: Exit code: 1", context)
            self.assertIn("failed targets are not validated", context)

    def test_repetitive_prior_summary_is_not_treated_as_plan_progress(self):
        with tempfile.TemporaryDirectory() as directory:
            plan = parse_request(Path(directory), "Implement main.py", session_id="work")
            repeated = "I will review the same files before editing them.\n</think>\n\n" * 20
            plan.record_progress([], response=repeated)
            context = planning_context(plan)
            self.assertIn("Host note: The previous assistant reply was repetitive", context)
            self.assertNotIn(repeated[:150], context)
            self.assertEqual(plan.steps[1].status, "pending")

    def test_continue_does_not_resume_stale_unrelated_active_plan(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            stale = parse_request(root, "clear", session_id="work")
            latest = parse_request(root, "Implement main.py", session_id="work")
            latest.record_progress([], response="Finished the task.")
            # The new progress rules do not accept a response alone as proof
            # of implementation. Mark this plan complete to represent a
            # genuinely finished newer task.
            for step in latest.steps:
                step.status = "complete"
            latest.refresh_status()
            latest.updated_at = (datetime.now(timezone.utc) + timedelta(seconds=5)).isoformat()
            from wallace_planning import save_plan
            save_plan(root, latest)
            resumed = resume_or_create(root, "Continue", session_id="work")
            self.assertNotEqual(resumed.plan_id, stale.plan_id)
            self.assertNotEqual(resumed.plan_id, latest.plan_id)

    def test_continue_selects_newest_active_plan_in_thread(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            stale = parse_request(root, "clear", session_id="work")
            latest = parse_request(root, "Implement main.py", session_id="work")
            latest.updated_at = (datetime.now(timezone.utc) + timedelta(seconds=5)).isoformat()
            from wallace_planning import save_plan
            save_plan(root, latest)
            resumed = resume_or_create(root, "Continue", session_id="work")
            self.assertEqual(resumed.plan_id, latest.plan_id)
            self.assertNotEqual(resumed.plan_id, stale.plan_id)

    def test_validation_before_edit_is_not_evidence_of_validation(self):
        with tempfile.TemporaryDirectory() as directory:
            plan = parse_request(Path(directory), "Implement main.py")
            plan.record_progress([
                {"id": "read", "name": "read_safe_file", "target": "main.py", "outcome": "ok", "ok": True},
                {"id": "test", "name": "run_tests", "target": "tests", "outcome": "Exit code: 0", "ok": True},
            ])
            self.assertEqual(plan.steps[2].status, "pending")
            plan.record_progress([
                {"id": "edit", "name": "replace_in_file", "target": "main.py", "outcome": "ok", "ok": True},
            ], response="Done")
            self.assertEqual(plan.steps[2].status, "pending")
            self.assertEqual(plan.status, "active")

    def test_threads_are_isolated_and_legacy_history_migrates_to_main(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            legacy_dir = root / ".wallace"
            legacy_dir.mkdir()
            (legacy_dir / "conversation.json").write_text(
                json.dumps([{"role": "user", "content": "legacy"}])
            )
            manager = ConversationThreadManager(root)
            main_store = manager.store("main")
            other = manager.create("Research")
            other_store = manager.store(other.thread_id)
            self.assertEqual(main_store.load()[0]["content"], "legacy")
            main_store.save([
                {"role": "user", "content": "main prompt"},
                {"role": "assistant", "content": "main only"},
            ])
            other_store.save([
                {"role": "user", "content": "other prompt"},
                {"role": "assistant", "content": "other only"},
            ])
            self.assertEqual(main_store.load()[-1]["content"], "main only")
            self.assertEqual(other_store.load()[-1]["content"], "other only")
            renamed = manager.rename(other.thread_id, "Research renamed")
            self.assertEqual(renamed.title, "Research renamed")
            self.assertEqual(len(manager.list_threads()), 2)
            self.assertEqual(manager.resolve("Research renamed").thread_id, other.thread_id)
            self.assertEqual(manager.resolve(other.thread_id).title, "Research renamed")
            with self.assertRaises(ValueError):
                manager.resolve("does-not-exist")
            removed = manager.remove("Research renamed")
            self.assertEqual(removed.thread_id, other.thread_id)
            self.assertEqual([record.thread_id for record in manager.list_threads()], ["main"])
            self.assertFalse(manager.conversation_path(other.thread_id).exists())
            with self.assertRaises(ValueError):
                manager.remove("main")

    def test_thread_commands_switch_and_delete_by_title(self):
        from main import _handle_thread_command

        with tempfile.TemporaryDirectory() as directory:
            manager = ConversationThreadManager(Path(directory))
            manager.ensure("main")
            created = manager.create("build_wallace")
            manager.store(created.thread_id).save([
                {"role": "user", "content": "hello"},
                {"role": "assistant", "content": "hi"},
            ])

            selected = _handle_thread_command("/switch build_wallace", manager, "main")
            self.assertEqual(selected, created.thread_id)
            self.assertEqual(len(manager.list_threads()), 2)

            selected = _handle_thread_command("/delete build_wallace", manager, selected)
            self.assertEqual(selected, "main")
            self.assertEqual([record.thread_id for record in manager.list_threads()], ["main"])
            self.assertFalse(manager.conversation_path(created.thread_id).exists())


if __name__ == "__main__":
    unittest.main()
