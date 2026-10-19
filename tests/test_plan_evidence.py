"""Host evidence must describe observed scope, without certifying feature claims."""

import tempfile
import unittest
from pathlib import Path

from wallace_planning import PlanSpec, PlanStep, create_plan, parse_request, planning_context


def event(call_id, name, target="", **metadata):
    return {"id": call_id, "name": name, "target": target, "ok": True, "outcome": "ok", **metadata}


class PlanEvidenceTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)

    def plan(self, request="Implement the requested behavior"):
        return parse_request(self.root, request)

    def test_unrelated_scoped_tests_do_not_validate_python_module(self):
        plan = self.plan()
        plan.record_progress([
            event("read", "read_safe_file", "src/payment.py"),
            event("edit", "write_safe_file", "src/payment.py"),
            event("test", "run_tests", "tests/test_unrelated.py"),
        ], response="Everything works.")
        self.assertEqual(plan.unchecked_targets(), ["src/payment.py"])
        self.assertEqual(plan.status, "active")
        plan.record_progress([event("suite", "run_tests", ".")], response="Full suite passed.")
        self.assertEqual(plan.status, "complete")

    def test_authorized_fix_is_not_misread_as_a_capability_question(self):
        plan = self.plan(
            "Fix counter.py so add(a, b) returns a + b. You are authorized to edit "
            "only counter.py. Read it first, then check its syntax with an available tool."
        )
        self.assertEqual(plan.intent, "fix")
        self.assertEqual(len(plan.steps), 4)

    def test_exact_test_file_and_host_coverage_mapping_have_explicit_scope(self):
        plan = self.plan()
        plan.record_progress([
            event("edit-code", "write_safe_file", "src/payment.py"),
            event("edit-test", "write_safe_file", "tests/test_payment.py"),
            event("test", "run_tests", "tests/test_payment.py", validation_targets=["src/payment.py"]),
        ])
        self.assertEqual(plan.unchecked_targets(), [])
        restored = PlanSpec.from_dict(plan.to_dict())
        self.assertEqual(restored.tool_history[-1]["validation_targets"], ["src/payment.py"])

    def test_source_only_completion_never_claims_functional_proof(self):
        plan = self.plan()
        plan.record_progress([
            event("read", "read_safe_file", "app.py"),
            event("edit", "write_safe_file", "app.py"),
            event("syntax", "check_python_syntax", "app.py"),
        ], response="The syntax check passed.")
        self.assertEqual(plan.status, "complete")
        self.assertIn("functional behavior was not tested", plan.verification_note())
        self.assertIn("Check limits:", planning_context(plan))
        self.assertNotIn("Working requested behavior", planning_context(plan))

    def test_plain_text_uses_post_write_readback_without_irrelevant_code_checks(self):
        plan = self.plan("Create notes.md with the requested meeting notes")
        plan.record_progress([
            event("inspect", "get_project_tree", "."),
            event("write", "write_safe_file", "notes.md"),
        ], response="Saved the notes.")
        self.assertEqual(plan.status, "active")
        plan.record_progress([event("readback", "read_safe_file", "notes.md")], response="Read the saved notes.")
        self.assertEqual(plan.status, "complete")
        self.assertIn("Saved text read back", plan.verification_note())

    def test_a_read_alone_does_not_satisfy_a_code_validation_request(self):
        plan = self.plan("Verify the code and run tests")
        plan.record_progress([event("read", "read_safe_file", "app.py")], response="Checked it.")
        self.assertEqual(plan.status, "active")

    def test_fabricated_test_success_is_replaced_with_observed_source_checks(self):
        from wallace_execution import verified_plan_response
        plan = self.plan()
        answer = verified_plan_response(plan, "All tests passed.", [
            event("read", "read_safe_file", "app.py"),
            event("edit", "write_safe_file", "app.py"),
            event("syntax", "check_source_file", "app.py"),
        ])
        self.assertNotIn("All tests passed", answer)
        self.assertIn("check_source_file", answer)
        self.assertIn("functional behavior was not tested", answer)

    def test_passing_syntax_does_not_erase_failed_tests_for_same_file(self):
        plan = self.plan()
        failed = event("failed", "run_tests", "tests/test_payment.py")
        failed["ok"] = False
        plan.record_progress([
            event("edit", "write_safe_file", "tests/test_payment.py"), failed,
            event("syntax", "check_python_syntax", "tests/test_payment.py"),
        ], response="Syntax passed.")
        self.assertEqual(plan.status, "active")
        self.assertEqual(plan.unresolved_failures[0]["name"], "run_tests")

    def test_command_and_external_mutations_without_scope_remain_unverified(self):
        for mutation in [
            event("command", "execute_command", "python change_files.py"),
            event("external", "external_update", "remote workspace", mutation=True),
            event("redirect", "execute_command", "cat template > app.py"),
            event("script", "execute_command", "python test_and_change.py"),
            event("nested", "execute_command", "find . -exec python change.py {} \\;"),
        ]:
            with self.subTest(name=mutation["name"], target=mutation["target"]):
                plan = self.plan("Please proceed")
                plan.record_progress([mutation], response="Done.")
                self.assertEqual(plan.status, "active")
                self.assertIn("without a recorded change scope", plan.incomplete_handoff())
                restored = PlanSpec.from_dict(plan.to_dict())
                restored.record_progress([mutation], response="Done.")
                self.assertEqual(len(restored.edit_evidence["unscoped_mutations"]), 1)

    def test_known_command_changes_need_checks_and_can_resolve_unknown_effects(self):
        plan = self.plan()
        plan.record_progress([
            event("read", "read_safe_file", "app.py"),
            event("unknown", "execute_command", "python change_files.py"),
        ])
        plan.record_progress([
            event("reconcile", "host_change_scope", mutation=False,
                  changed_paths=["app.py"], resolved_mutation_ids=["unknown"]),
            event("suite", "run_tests", "."),
        ], response="Changes inspected and tests passed.")
        self.assertEqual(plan.status, "complete")
        self.assertFalse(plan.edit_evidence.get("unscoped_mutations"))

    def test_multiple_feature_request_uses_observed_stages_not_feature_counters(self):
        plan = self.plan("Implement features: search, export, and validation.")
        self.assertEqual(len(plan.steps), 4)
        plan.record_progress([event("edit", "write_safe_file", "app.py")])
        self.assertEqual(plan.steps[1].status, "complete")
        self.assertIn("Successful file edits recorded", plan.steps[1].result)
        self.assertNotIn("search", plan.steps[1].result)
        self.assertEqual(plan.steps[-2].status, "pending")

    def test_old_feature_plan_round_trips_and_collapses_without_false_feature_claims(self):
        plan = create_plan(self.root, "Implement features: search, export", intent="implement", steps=[
            PlanStep("step-01", "Inspect relevant context"),
            PlanStep("step-02", "Implement search"),
            PlanStep("step-03", "Implement export"),
            PlanStep("step-04", "Validate the result"),
            PlanStep("step-05", "Summarize and hand off"),
        ])
        restored = PlanSpec.from_dict(plan.to_dict())
        restored.record_progress([event("edit", "write_safe_file", "app.py")])
        self.assertEqual(len(restored.steps), 4)
        self.assertEqual(restored.steps[1].title, "Apply requested changes")
        self.assertEqual(restored.steps[-2].status, "pending")

    def test_incremental_tool_events_are_idempotent_and_reopen_only_changed_checks(self):
        plan = self.plan()
        actions = [event("read", "read_safe_file", "app.py"),
                   event("edit", "write_safe_file", "app.py"),
                   event("check", "check_python_syntax", "app.py")]
        for action in actions:
            plan.record_progress([action])
        self.assertEqual(plan.status, "active")
        plan.record_progress(actions, response="Checks passed.")
        self.assertEqual(plan.status, "complete")
        self.assertEqual(len(plan.tool_history), 3)
        plan.record_progress([event("edit-again", "write_safe_file", "app.py")])
        self.assertEqual(plan.unchecked_targets(), ["app.py"])
        self.assertEqual(plan.status, "active")


if __name__ == "__main__":
    unittest.main()
