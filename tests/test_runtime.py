import unittest

from simulator.runtime import RankClock, Shape, StepModel


class RuntimeTest(unittest.TestCase):
    def test_late_rank_controls_release(self) -> None:
        clock = RankClock(2)
        self.assertEqual(clock.complete(0, [10, 30]), 30)
        self.assertEqual(clock.complete(30, [40, 5]), 70)

    def test_missing_rank_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "Every rank"):
            RankClock(2).complete(0, [10])

    def test_past_release_rejected(self) -> None:
        clock = RankClock(2)
        clock.complete(0, [10, 20])
        with self.assertRaisesRegex(ValueError, "frontier"):
            clock.complete(10, [5, 5])

    def test_interpolation_and_domain_rejection(self) -> None:
        samples = [{"shape": {"phase": "decode", "batch": b, "context": c},
                    "duration_ns": b * c} for b in (8, 32) for c in (64, 128)]
        model = StepModel(samples)
        self.assertEqual(model.predict(Shape("decode", 16, 96)), 1536)
        with self.assertRaisesRegex(ValueError, "domain"):
            model.predict(Shape("decode", 64, 96))
        with self.assertRaisesRegex(ValueError, "domain"):
            model.predict(Shape("decode", 16, 256))
        with self.assertRaisesRegex(ValueError, "Uncalibrated"):
            model.predict(Shape("mixed", 16, 96))


if __name__ == "__main__":
    unittest.main()
