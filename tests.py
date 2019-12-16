import os
from unittest import TestCase
from tempfile import TemporaryDirectory
from luigi import build

from pset.tasks.sambanis_pipeline import SambanisRFCV


class TestSambanisRF(TestCase):
    """Tests known values for fitting a random forest to Sambanis data
    """

    def test_with_good_params(self):
        """With 100 estimators, a random forest is known to be able to achieve close to 0.9 AUC, but not 1.0"""
        with TemporaryDirectory() as td:
            build(
                [SambanisRFCV(output_path=td, model_params={"n_estimators": 100})],
                local_scheduler=True,
            )
            with open(os.path.join(td, "auc.txt")) as f:
                result = float(f.read())
        self.assertAlmostEqual(result, 0.9, delta=0.05)

    def test_with_bad_params(self):
        """With only 1 estimator, a random forest should perform much worse than with 100"""
        with TemporaryDirectory() as td:
            build(
                [SambanisRFCV(output_path=td, model_params={"n_estimators": 1})],
                local_scheduler=True,
            )
            with open(os.path.join(td, "auc.txt")) as f:
                result = float(f.read())
        print(result)
        self.assertLess(result, 0.8)
