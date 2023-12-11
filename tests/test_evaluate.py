import unittest
import numpy as np

import pandas as pd
import infectio.evaluate


class TestTimeseriesPoint2distributionDistance(unittest.TestCase):
    def setUp(self) -> None:
        self.refdf = pd.DataFrame(
            {
                "t": np.linspace(5, 9, 5),
                "mean_count": np.linspace(20, 30, 5),
                "std_count": np.linspace(2, 3, 5),
            }
        )
        self.targetdf = pd.DataFrame(
            {
                "t": np.linspace(0, 10, 11),
                "count": np.linspace(10, 32, 11),
            }
        )

    def test_correctness(self):
        result = infectio.evaluate.evaluate_simulation_against_reference(
            self.refdf, self.targetdf, "mean_count", "std_count", "count"
        )
        self.assertAlmostEqual(
            sum(result), 21.5, places=1
        )  # TODO: Needs to be checked if this is correct

        self.assertEqual(len(result), 1)

    def test_slight_alignment(self):
        modified_refdf = self.refdf.copy()
        modified_refdf["t"] += 0.1
        result = infectio.evaluate.evaluate_simulation_against_reference(
            modified_refdf, self.targetdf, "mean_count", "std_count", "count"
        )
        self.assertAlmostEqual(
            sum(result), 21.5, places=1
        )  # TODO: Needs to be checked if this is correct
        self.assertEqual(len(result), 1)


if __name__ == "__main__":
    unittest.main()
