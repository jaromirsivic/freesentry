import unittest

from server.speedhistogram import SpeedHistogram


class SpeedHistogramValidationTests(unittest.TestCase):
    def test_rejects_zero_max_forward_seconds(self):
        with self.assertRaisesRegex(ValueError, "forwardSeconds greater than 0"):
            SpeedHistogram(
                speed_histogram=[
                    {"pwmMultiplier": 0, "forwardSeconds": 0, "reverseSeconds": 0},
                    {"pwmMultiplier": 1, "forwardSeconds": 0, "reverseSeconds": 10},
                ]
            )

    def test_rejects_zero_max_reverse_seconds(self):
        with self.assertRaisesRegex(ValueError, "reverseSeconds greater than 0"):
            SpeedHistogram(
                speed_histogram=[
                    {"pwmMultiplier": 0, "forwardSeconds": 0, "reverseSeconds": 0},
                    {"pwmMultiplier": 1, "forwardSeconds": 10, "reverseSeconds": 0},
                ]
            )


if __name__ == "__main__":
    unittest.main()
