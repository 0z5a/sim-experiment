import unittest

from tools.report_crossrank import ERROR_METRICS, external_ids, prediction_hits


class RequestMappingTest(unittest.TestCase):
    def test_prediction_hits_use_individual_trials_and_inclusive_threshold(self) -> None:
        cases = [{key: value for key in ERROR_METRICS} for value in (-10.0, 10.0, 10.01)]
        cases[1]["ttft_error_pct"] = -10.01
        result = prediction_hits(cases)
        self.assertEqual(result["hits"]["duration_error_pct"], 2)
        self.assertEqual(result["hits"]["ttft_error_pct"], 1)
        self.assertEqual(result["hits"]["all_metrics"], 1)
        self.assertEqual(result["trials"], 3)

    def test_rank_suffixes_do_not_change_membership(self) -> None:
        left = [{"ids": ["0-1234abcd", "1-5678abcd"]}]
        right = [{"ids": ["0-abcd1234", "1-abcd5678"]}]
        self.assertEqual(external_ids(left, 2), external_ids(right, 2))
        self.assertEqual(external_ids(left, 2), [["0", "1"]])

    def test_collision_and_missing_request_rejected(self) -> None:
        with self.assertRaises(ValueError):
            external_ids([{"ids": ["0-1234abcd", "0-5678abcd"]}], 2)

    def test_unrecognized_format_rejected(self) -> None:
        with self.assertRaises(ValueError):
            external_ids([{"ids": ["0-other"]}], 1)
