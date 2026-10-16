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
    continuation_instruction,
    create_plan,
    load_plan,
    parse_request,
    planning_context,
    resume_or_create,
)


class PlanningTests(unittest.TestCase):
    def test_diagnose_requests_require_inspection_step(self):
        with tempfile.TemporaryDirectory() as directory:
            plan = parse_request(Path(directory), "Diagnose why get_git_status returned nothing / errored.")
        self.assertEqual(plan.intent, "review")
        self.assertEqual(plan.steps[0].title, "Inspect relevant context")
        self.assertEqual(plan.steps[-1].title, "Summarize and hand off")

    def test_detailed_user_correction_survives_terse_continue(self):
        correction = (
            "Continue the unfinished deletion of test_site. The directory is already "
            "absent; do not repeat the removal. Run the full tests and report."
        )
        history = [
            {"role": "user", "content": correction},
            {"role": "assistant", "content": "The plan is unfinished."},
            {"role": "user", "content": "Continue executing my request."},
        ]
        self.assertEqual(
            continuation_instruction("Continue executing my request.", history),
            correction,
        )

    def test_legacy_invalid_syntax_call_does_not_block_real_test_check(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            plan = parse_request(root, "Verify the code in test_site and run tests")
            plan.record_progress([{
                "id": "typo", "name": "check_python_syntax",
                "target": "tests/test_core_safety", "outcome": "failed", "ok": False,
            }, {
                "id": "old-suite", "name": "run_tests", "target": "",
                "outcome": "Exit code: 5", "ok": False,
            }])
            self.assertEqual([item["name"] for item in plan.unresolved_failures], ["run_tests"])
            context = planning_context(plan)
            self.assertIn("Unresolved checks", context)
            self.assertIn("run_tests", context)
            plan.record_progress([{
                "id": "real-suite", "name": "run_tests", "target": ".",
                "outcome": "Exit code: 0", "ok": True,
            }], response="The full tests passed.")
            self.assertEqual(plan.unresolved_failures, [])
            self.assertEqual(plan.status, "complete")

    def test_legacy_delete_context_forbids_repeating_removal(self):
        with tempfile.TemporaryDirectory() as directory:
            plan = parse_request(Path(directory), "Proceed with deleting all test-site files.")
            plan.intent = "test"  # Legacy parser behavior retained in old plans.
            plan.record_progress([{
                "id": "rm", "name": "execute_command", "target": "rm -rf test_site",
                "outcome": "Exit code: 0", "ok": True,
            }])
            context = planning_context(
                plan, latest_instruction="Continue: test_site is absent; do not delete again."
            )
            self.assertIn("Do not remove the target again", context)
            self.assertIn("Latest detailed user instruction", context)

    def test_resume_repairs_old_invalid_invocation_without_erasing_test_failure(self):
        from wallace_planning import save_plan

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            plan = parse_request(root, "Verify code in test_site", session_id="work")
            plan.unresolved_failures = [
                {"name": "check_python_syntax", "target": "tests/test_core_safety", "outcome": "failed", "diagnostic": ""},
                {"name": "run_tests", "target": "", "outcome": "Exit code: 5", "diagnostic": ""},
            ]
            save_plan(root, plan)
            resumed = resume_or_create(root, "Continue executing my request.", session_id="work")
            self.assertEqual(resumed.plan_id, plan.plan_id)
            self.assertEqual([item["name"] for item in resumed.unresolved_failures], ["run_tests"])

    def test_full_suite_success_clears_legacy_default_target_failure(self):
        with tempfile.TemporaryDirectory() as directory:
            plan = parse_request(Path(directory), "Verify the project tests")
            plan.record_progress([{
                "id": "old", "name": "run_tests", "target": "",
                "outcome": "Exit code: 5", "ok": False,
            }])
            self.assertEqual(len(plan.unresolved_failures), 1)
            plan.record_progress([{
                "id": "pass", "name": "run_tests", "target": ".",
                "outcome": "Exit code: 0", "ok": True,
            }])
            self.assertEqual(plan.unresolved_failures, [])

    def test_line_editor_counts_as_a_real_edit(self):
        with tempfile.TemporaryDirectory() as directory:
            plan = parse_request(Path(directory), "Fix example.py")
            plan.record_progress([{
                "id": "line-edit", "name": "edit_file_lines", "target": "example.py",
                "outcome": "ok", "ok": True,
            }])
            self.assertIn("example.py", plan.unchecked_targets())
            plan.record_progress([{
                "id": "syntax", "name": "check_python_syntax", "target": "example.py",
                "outcome": "ok", "ok": True,
            }])
            self.assertEqual(plan.unchecked_targets(), [])

    def test_read_only_site_question_is_not_misread_as_creation_or_testing(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for request in (
                "Look at the test site you created and tell me what it features.",
                "What is the website about?",
            ):
                plan = parse_request(root, request)
                self.assertEqual(plan.intent, "review")
                self.assertEqual(len(plan.steps), 2)

    def test_deletion_needs_absence_git_and_full_tests_not_just_a_listing(self):
        from main import verified_plan_response

        with tempfile.TemporaryDirectory() as directory:
            plan = parse_request(Path(directory), "Delete the site.")
            self.assertEqual(plan.intent, "delete")
            self.assertEqual(len(plan.steps), 4)
            reply = verified_plan_response(plan, "Site fully removed.", [{
                "id": "read", "name": "get_project_tree", "target": "test_site", "outcome": "ok", "ok": True,
            }])
            self.assertIn("No successful deletion command", reply)
            reply = verified_plan_response(plan, "Site fully removed.", [{
                "id": "rm", "name": "execute_command", "target": "rm -rf ./test_site", "outcome": "Exit code: 0", "ok": True,
            }])
            self.assertIn("Target absence", reply)
            reply = verified_plan_response(plan, "Site fully removed.", [
                {"id": "wrong-absence", "name": "execute_command", "target": "test ! -e ./unrelated", "outcome": "Exit code: 0", "ok": True},
                {"id": "wrong-git", "name": "get_git_status", "target": "", "outcome": "Exit code: 0", "ok": True},
                {"id": "wrong-suite", "name": "run_tests", "target": ".", "outcome": "Exit code: 0", "ok": True},
            ])
            self.assertIn("Target absence", reply)
            reply = verified_plan_response(plan, "Site fully removed.", [
                {"id": "absent", "name": "execute_command", "target": "test ! -e ./test_site", "outcome": "Exit code: 0", "ok": True},
                {"id": "git", "name": "get_git_status", "target": "", "outcome": "Exit code: 0", "ok": True},
                {"id": "failed-suite", "name": "run_tests", "target": ".", "outcome": "Exit code: 1", "ok": False},
            ])
            self.assertIn("Unresolved run_tests failure", reply)
            self.assertEqual(plan.status, "active")
            reply = verified_plan_response(plan, "Site removed and checks passed.", [{
                "id": "suite", "name": "run_tests", "target": ".", "outcome": "Exit code: 0", "ok": True,
            }])
            self.assertEqual(reply, "Site removed and checks passed.")
            self.assertEqual(plan.status, "complete")

    def test_code_verification_cannot_finish_after_reads_or_failed_checks(self):
        from main import verified_plan_response

        with tempfile.TemporaryDirectory() as directory:
            plan = parse_request(Path(directory), "Verify the code in test_site/app.ts and test it")
            self.assertEqual(plan.intent, "test")
            self.assertEqual(len(plan.steps), 3)
            response = verified_plan_response(plan, "The site is verified.", [{
                "id": "read", "name": "read_safe_file", "target": "test_site/app.ts",
                "outcome": "ok", "ok": True,
            }])
            self.assertEqual(plan.status, "active")
            self.assertIn("Required code checks or tests have not passed", response)
            response = verified_plan_response(plan, "The site is verified.", [{
                "id": "failed-check", "name": "check_source_file", "target": "test_site/app.ts",
                "outcome": "Exit code: 1", "diagnostic": "TypeScript type-check failed (TS2584)",
                "ok": False,
            }])
            self.assertEqual(plan.status, "active")
            self.assertIn("TS2584", response)
            self.assertNotIn("The site is verified", response)
            verified_plan_response(plan, "The site passed its checks.", [{
                "id": "passed-check", "name": "check_source_file", "target": "test_site/app.ts",
                "outcome": "Exit code: 0", "ok": True,
            }])
            self.assertEqual(plan.status, "complete")

    def test_non_code_testing_request_does_not_require_source_checks(self):
        with tempfile.TemporaryDirectory() as directory:
            plan = parse_request(Path(directory), "Test me on history")
            self.assertEqual(plan.intent, "test")
            self.assertEqual(len(plan.steps), 2)

    def test_verification_followup_resumes_unfinished_deletion_check(self):
        from main import verified_plan_response

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            deletion = parse_request(root, "Proceed with deleting all test-site files.", session_id="work")
            self.assertEqual(len(deletion.steps), 4)
            deletion.record_progress([
                {"id": "read", "name": "get_project_tree", "target": "test_site", "outcome": "ok", "ok": True},
                {"id": "delete", "name": "execute_command", "target": "rm -rf ./test_site", "outcome": "Exit code: 0", "ok": True},
            ], response="Deleted the site.")
            self.assertEqual(deletion.status, "active")
            from wallace_planning import save_plan
            save_plan(root, deletion)
            followup = resume_or_create(root, "Verify its completion.", session_id="work")
            self.assertEqual(followup.plan_id, deletion.plan_id)

            # A separate, completed verification reply must not bury the still-active work.
            shallow = parse_request(root, "Verify its completion.", session_id="work")
            for step in shallow.steps:
                step.status = "complete"
            shallow.refresh_status()
            save_plan(root, shallow)
            continued = resume_or_create(root, "Continue finishing the deletion.", session_id="work")
            self.assertEqual(continued.plan_id, deletion.plan_id)

            response = verified_plan_response(continued, "Everything is cleaned up.", [{
                "id": "listing", "name": "get_project_tree", "target": "", "outcome": "ok", "ok": True,
            }])
            self.assertIn("Target absence, Git status, and full project tests", response)

    def test_review_plan_cannot_finish_with_unvalidated_edits(self):
        from main import verified_plan_response

        with tempfile.TemporaryDirectory() as directory:
            plan = parse_request(Path(directory), "Review test_site/index.html")
            response = verified_plan_response(plan, "The page is ready.", [
                {"id": "read", "name": "read_safe_file", "target": "test_site/index.html", "outcome": "ok", "ok": True},
                {"id": "edit", "name": "replace_in_file", "target": "test_site/index.html", "outcome": "ok", "ok": True},
            ])
            self.assertEqual(plan.status, "active")
            self.assertIn("Files needing matching validation", response)
            verified_plan_response(plan, "The page is checked.", [{
                "id": "check", "name": "check_source_file", "target": "test_site/index.html",
                "outcome": "Exit code: 0", "ok": True,
            }])
            self.assertEqual(plan.status, "complete")

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
                {"id": "edit-2", "name": "replace_in_file", "target": "translator.py", "outcome": "ok", "ok": True},
                {"id": "test-2", "name": "run_tests", "target": "tests", "outcome": "ok", "ok": True},
            ], response="Implemented and tested translation.")
            self.assertEqual([step.status for step in plan.steps[1:-2]], ["complete", "complete", "pending"])
            plan.record_progress([
                {"id": "edit-3", "name": "replace_in_file", "target": "validator.py", "outcome": "ok", "ok": True},
                {"id": "test-3", "name": "run_tests", "target": "tests", "outcome": "ok", "ok": True},
            ], response="Implemented and tested output validation.")
            self.assertEqual(plan.status, "complete")

    def test_tool_registry_edit_requires_contract_check_after_latest_edit(self):
        with tempfile.TemporaryDirectory() as directory:
            plan = parse_request(Path(directory), "Implement a new tool in tools.py")
            plan.record_progress([
                {"id": "read", "name": "read_safe_file", "target": "tools.py", "outcome": "ok", "ok": True},
                {"id": "edit", "name": "replace_in_file", "target": "tools.py", "outcome": "ok", "ok": True},
                {"id": "tests", "name": "run_tests", "target": "tests", "outcome": "Exit code: 0", "ok": True},
            ], response="The tests pass.")
            self.assertEqual(plan.status, "active")
            plan.record_progress([{
                "id": "contract", "name": "verify_tool_contract", "target": "new_tool",
                "outcome": "ok", "ok": True,
            }], response="The new tool is registered and its test passes.")
            self.assertEqual(plan.status, "complete")
            plan.record_progress([{
                "id": "edit-again", "name": "replace_in_file", "target": "tools.py",
                "outcome": "ok", "ok": True,
            }], response="Adjusted the tool.")
            self.assertEqual(plan.status, "active")

    def test_edit_proof_survives_short_journal_and_restart(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            plan = parse_request(root, "Implement a new tool in tools.py", session_id="work")
            activity = [
                {"id": "read", "name": "read_safe_file", "target": "tools.py", "outcome": "ok", "ok": True},
                {"id": "edit", "name": "replace_in_file", "target": "tools.py", "outcome": "ok", "ok": True},
                {"id": "contract", "name": "verify_tool_contract", "target": "new_tool", "outcome": "ok", "ok": True},
            ]
            activity.extend({
                "id": f"extra-{number}", "name": "search_in_files", "target": "tests",
                "outcome": "ok", "ok": True,
            } for number in range(20))
            plan.record_progress(activity, error="malformed tool call")
            self.assertNotIn("edit", {item["id"] for item in plan.tool_history})
            from wallace_planning import save_plan
            save_plan(root, plan)
            restored = load_plan(root, "work", plan.plan_id)
            self.assertEqual(restored.edit_evidence["target"], "tools.py")
            self.assertIn("verify_tool_contract", restored.edit_evidence["validations"])
            restored.record_progress(activity, response="The tool is verified.")
            self.assertEqual(restored.status, "complete")
            self.assertEqual(len(restored.seen_tool_call_ids), len(activity))

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

    def test_try_again_resumes_unfinished_work_in_the_same_thread(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            unfinished = parse_request(root, "Fix test_site/app.ts", session_id="work")
            other = parse_request(root, "Fix other.py", session_id="other")
            for step in other.steps:
                step.status = "complete"
            other.refresh_status()
            from wallace_planning import save_plan
            save_plan(root, other)
            for request in ("Try again.", "Please try again!", "retry that"):
                resumed = resume_or_create(root, request, session_id="work")
                self.assertEqual(resumed.plan_id, unfinished.plan_id)

    def test_continue_skips_legacy_no_op_retry_plan(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            unfinished = parse_request(root, "Fix test_site/app.ts", session_id="work")
            noop = parse_request(root, "Try again.", session_id="work")
            noop.record_progress([], response="Done")
            from wallace_planning import save_plan
            save_plan(root, noop)
            resumed = resume_or_create(root, "Continue", session_id="work")
            self.assertEqual(resumed.plan_id, unfinished.plan_id)

    def test_retry_skips_active_legacy_no_op_retry_plan(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            unfinished = parse_request(root, "Fix test_site/app.ts", session_id="work")
            noop = parse_request(root, "Try again.", session_id="work")
            resumed = resume_or_create(root, "Try again.", session_id="work")
            self.assertNotEqual(noop.plan_id, resumed.plan_id)
            self.assertEqual(resumed.plan_id, unfinished.plan_id)

    def test_generic_request_cannot_complete_with_unchecked_file_edits(self):
        from main import verified_plan_response

        with tempfile.TemporaryDirectory() as directory:
            plan = parse_request(Path(directory), "Do it", session_id="work")
            answer = verified_plan_response(plan, "The CSS passed validation.", [{
                "id": "edit", "name": "write_safe_file", "target": "test_site/styles.css",
                "outcome": "ok", "ok": True,
            }])
            self.assertEqual(plan.status, "active")
            self.assertIn("test_site/styles.css", answer)
            self.assertNotIn("passed validation", answer)
            verified = verified_plan_response(plan, "CSS checked.", [{
                "id": "check", "name": "check_source_file", "target": "test_site/styles.css",
                "outcome": "ok", "ok": True,
            }])
            self.assertEqual(verified, "CSS checked.")
            self.assertEqual(plan.status, "complete")

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

    def test_unrelated_python_check_and_failed_web_checks_cannot_complete_site(self):
        from main import verified_plan_response

        with tempfile.TemporaryDirectory() as directory:
            plan = parse_request(Path(directory), "Build a website in test_site")
            activity = [
                {"id": "read", "name": "read_safe_file", "target": "README.md", "outcome": "ok", "ok": True},
                {"id": "html", "name": "write_safe_file", "target": "test_site/index.html", "outcome": "ok", "ok": True},
                {"id": "ts", "name": "write_safe_file", "target": "test_site/app.ts", "outcome": "ok", "ok": True},
                {"id": "sh", "name": "write_safe_file", "target": "test_site/build.sh", "outcome": "ok", "ok": True},
                {"id": "python", "name": "check_python_syntax", "target": "dependency_analysis.py", "outcome": "ok", "ok": True},
                {"id": "deno", "name": "execute_command", "target": "cd test_site && deno check app.ts", "outcome": "Exit code: 1", "ok": False},
                {"id": "replace", "name": "replace_in_file", "target": "test_site/app.ts", "outcome": "failed", "diagnostic": "replacement anchor not found; re-read the file", "ok": False},
            ]
            handoff = verified_plan_response(plan, "The TypeScript compiles; the site works.", activity)
            self.assertEqual(plan.status, "active")
            self.assertIn("has not verified", handoff)
            self.assertNotIn("compiles; the site works", handoff)
            self.assertEqual(set(plan.unchecked_targets()), {
                "test_site/index.html", "test_site/app.ts", "test_site/build.sh",
            })
            self.assertEqual(len(plan.unresolved_failures), 2)
            self.assertIn("replacement anchor not found", planning_context(plan))
            self.assertIn("Files still needing matching checks", planning_context(plan))

    def test_every_changed_web_source_needs_its_own_check(self):
        from main import verified_plan_response

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            plan = parse_request(root, "Build a JavaScript website")
            activity = [
                {"id": "read", "name": "read_safe_file", "target": "README.md", "outcome": "ok", "ok": True},
                {"id": "ts", "name": "write_safe_file", "target": "site/app.ts", "outcome": "ok", "ok": True},
                {"id": "js", "name": "write_safe_file", "target": "site/app.js", "outcome": "ok", "ok": True},
                {"id": "check-ts", "name": "check_source_file", "target": "site/app.ts", "outcome": "Exit code: 0", "ok": True},
            ]
            self.assertIn("site/app.js", verified_plan_response(plan, "Done", activity))
            self.assertEqual(plan.status, "active")
            from wallace_planning import load_plan, save_plan
            save_plan(root, plan)
            restored = load_plan(root, plan.session_id, plan.plan_id)
            self.assertEqual(restored.unchecked_targets(), ["site/app.js"])
            result = verified_plan_response(restored, "Done", [{
                "id": "check-js", "name": "check_source_file", "target": "site/app.js",
                "outcome": "Exit code: 0", "ok": True,
            }])
            self.assertEqual(result, "Done")
            self.assertEqual(restored.status, "complete")

    def test_failed_edit_and_check_require_actual_recovery(self):
        with tempfile.TemporaryDirectory() as directory:
            plan = parse_request(Path(directory), "Fix site/app.ts")
            plan.record_progress([
                {"id": "read", "name": "read_safe_file", "target": "site/app.ts", "outcome": "ok", "ok": True},
                {"id": "edit", "name": "replace_in_file", "target": "site/app.ts", "outcome": "ok", "ok": True},
                {"id": "bad-check", "name": "check_source_file", "target": "site/app.ts", "outcome": "Exit code: 1", "ok": False},
                {"id": "bad-edit", "name": "replace_in_file", "target": "site/app.ts", "outcome": "failed", "ok": False},
            ], response="Fixed")
            self.assertEqual(plan.status, "active")
            plan.record_progress([{
                "id": "good-edit", "name": "replace_in_file", "target": "site/app.ts",
                "outcome": "ok", "ok": True,
            }], response="Fixed")
            self.assertEqual([item["name"] for item in plan.unresolved_failures], ["check_source_file"])
            plan.record_progress([{
                "id": "good-check", "name": "check_source_file", "target": "site/app.ts",
                "outcome": "Exit code: 0", "ok": True,
            }], response="Fixed and verified")
            self.assertEqual(plan.unresolved_failures, [])
            self.assertEqual(plan.status, "complete")

    def test_later_edit_reopens_only_its_own_validation(self):
        with tempfile.TemporaryDirectory() as directory:
            plan = parse_request(Path(directory), "Build a web page")
            plan.record_progress([
                {"id": "read", "name": "read_safe_file", "target": "README.md", "outcome": "ok", "ok": True},
                {"id": "js", "name": "write_safe_file", "target": "site/app.js", "outcome": "ok", "ok": True},
                {"id": "css", "name": "write_safe_file", "target": "site/styles.css", "outcome": "ok", "ok": True},
                {"id": "check-js", "name": "check_source_file", "target": "site/app.js", "outcome": "ok", "ok": True},
                {"id": "check-css", "name": "check_source_file", "target": "site/styles.css", "outcome": "ok", "ok": True},
            ], response="Done")
            self.assertEqual(plan.status, "complete")
            plan.record_progress([{
                "id": "edit-css", "name": "replace_in_file", "target": "site/styles.css",
                "outcome": "ok", "ok": True,
            }], response="Updated CSS")
            self.assertEqual(plan.status, "active")
            self.assertEqual(plan.unchecked_targets(), ["site/styles.css"])
            self.assertEqual(plan.steps[-1].status, "pending")
            plan.record_progress([{
                "id": "recheck-css", "name": "check_source_file", "target": "site/styles.css",
                "outcome": "ok", "ok": True,
            }], response="Updated CSS and checked")
            self.assertEqual(plan.status, "complete")

    def test_failed_compile_command_needs_retry_and_source_check(self):
        with tempfile.TemporaryDirectory() as directory:
            plan = parse_request(Path(directory), "Build app.ts")
            command = "deno check app.ts"
            plan.record_progress([
                {"id": "read", "name": "read_safe_file", "target": "README.md", "outcome": "ok", "ok": True},
                {"id": "edit", "name": "write_safe_file", "target": "app.ts", "outcome": "ok", "ok": True},
                {"id": "failed", "name": "execute_command", "target": command, "outcome": "Exit code: 1", "ok": False},
            ], response="Compiled")
            self.assertEqual(len(plan.unresolved_failures), 1)
            plan.record_progress([{
                "id": "recheck", "name": "execute_command", "target": command,
                "outcome": "Exit code: 0", "ok": True,
            }], response="Compiled")
            self.assertEqual(plan.unresolved_failures, [])
            self.assertEqual(plan.unchecked_targets(), ["app.ts"])
            plan.record_progress([{
                "id": "source", "name": "check_source_file", "target": "app.ts",
                "outcome": "Exit code: 0", "ok": True,
            }], response="Checked")
            self.assertEqual(plan.status, "complete")

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
