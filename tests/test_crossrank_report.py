import unittest

from tools.report_crossrank import external_ids


class RequestMappingTest(unittest.TestCase):
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
