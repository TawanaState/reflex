import unittest
from types import SimpleNamespace

from experiments.benchmark_paired_bfcl import score, strict_fields
from experiments.summarize_paired_bfcl import summarize


class StrictFieldScoreTest(unittest.TestCase):
    def test_alternatives_and_optional_omission(self):
        gold = {"id": [4, 5], "mode": ["", "fast"]}
        self.assertTrue(strict_fields({"id": 5}, gold))
        self.assertTrue(strict_fields({"id": 4, "mode": "fast"}, gold))
        self.assertFalse(strict_fields({"id": "5"}, gold))
        self.assertFalse(strict_fields({"id": 4, "other": 1}, gold))
        self.assertFalse(strict_fields({}, gold))

    def test_empty_call_is_exact(self):
        self.assertTrue(strict_fields({}, {}))
        self.assertFalse(strict_fields({"x": 1}, {}))

    def test_no_tool_scores_executable_api_call_only(self):
        row = {"arg_bucket": "no_tool"}
        self.assertTrue(score(SimpleNamespace(tool_calls=None), row)["no_tool_ok"])
        call = [{"function": {"name": "irrelevant", "arguments": "{}"}}]
        self.assertFalse(score(SimpleNamespace(tool_calls=call), row)["no_tool_ok"])

    def test_mixed_summary_handles_no_tool_name_accuracy(self):
        call = {"id": "call", "policy": "base", "bucket": "atomic", "source": "BFCL_v4_live_multiple",
                "latency_ms": 100, "strict_call_ok": True, "name_ok": True, "no_tool_ok": None,
                "forward_passes_estimate": 3, "early_exit": False, "pred_tool": "mute",
                "pred_args": {}, "content": None, "execution_path": "NATIVE_TOOL_CALL"}
        direct = dict(call, id="direct", bucket="no_tool", source="BFCL_v4_irrelevance",
                      strict_call_ok=None, name_ok=None, no_tool_ok=True, pred_tool=None,
                      pred_args=None, content="No tool applies", execution_path="NATIVE_DIRECT_RESPONSE")
        result = summarize([call, dict(call, policy="reflex_1"),
                            direct, dict(direct, policy="reflex_1")])
        self.assertEqual(result["groups"]["overall"]["base"]["name_accuracy"], 1.0)
        self.assertEqual(result["groups"]["no_tool"]["base"]["name_accuracy"], None)

    def test_summary_uses_complete_pairs_only(self):
        base = {"id": "a", "policy": "base", "bucket": "atomic",
                "latency_ms": 100, "strict_call_ok": True, "name_ok": True,
                "no_tool_ok": None, "forward_passes_estimate": 3, "early_exit": False,
                "pred_tool": "mute", "pred_args": {}, "content": None,
                "execution_path": "NATIVE_TOOL_CALL"}
        reflex = dict(base, policy="reflex_1", latency_ms=60,
                      forward_passes_estimate=1, early_exit=True)
        incomplete = dict(base, id="b")
        result = summarize([base, reflex, incomplete])
        self.assertEqual(result["complete_pairs"], 1)
        self.assertEqual(result["groups"]["atomic"]["paired_mean_saved_ms"], 40)
        self.assertEqual(result["groups"]["atomic"]["reflex_1"]["early_exits"], 1)


if __name__ == "__main__":
    unittest.main()
