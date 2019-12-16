import os
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from luigi import ExternalTask, LocalTarget

from .fold_data import SplitFoldsBase
from .fit_and_predict import FitPredictOnFoldBase
from .cross_validation import CrossValidateBase
from .hyperparameter_tuning import RandomForestTuningBase


class FearonLaitinData(ExternalTask):
    """Loads the original data via a hard-coded local path"""

    DATA_PATH = os.path.abspath("data")
    DATA_FILENAME = "fearon_laitin.csv"

    def output(self):
        return LocalTarget(os.path.join(self.DATA_PATH, self.DATA_FILENAME))


class FearonLaitinSplitFolds(SplitFoldsBase):
    """Splits the Fearon & Laitin data into k folds"""

    OUTPUT_NAME = "fearon_laitin"
    Y_COL = "onset"
    FEATURE_COLS = [
        "warl",
        "gdpenl",
        "lpop",
        "lmtnest",
        "ncontig",
        "Oil",
        "nwstate",
        "instab",
        "polity2l",
        "ethfrac",
        "relfrac",
    ]

    def requires(self):
        return FearonLaitinData()


class FearonLaitinRandomForest(FitPredictOnFoldBase):
    """Random Forest classifier for FearonLaitin data"""

    CLASSIFIER = RandomForestClassifier

    def requires(self):
        return FearonLaitinSplitFolds()


class FearonLaitinRFCV(CrossValidateBase):
    """Manages cross-validation for the FearonLaitinRandomForest classifier"""

    ModelTask = FearonLaitinRandomForest


class FearonLaitinRFTuning(RandomForestTuningBase):
    """Uses FearonLaitinRFCV to perform cross-validation while tuning the `n_estimators` parameter"""

    CrossValidationTask = FearonLaitinRFCV
    NAME = "Fearon & Laitin Dataset"
    n_estimator_values = range(5, 105, 5)


class FearonLaitinAdaboost(FearonLaitinRandomForest):
    """Adaboost classifier for FearonLaitin data"""

    CLASSIFIER = AdaBoostClassifier


class FearonLaitinAdaboostCV(CrossValidateBase):
    """Cross-validation for the Adaboost classifier"""

    ModelTask = FearonLaitinAdaboost


class FearonLaitinAdaboostTuning(FearonLaitinRFTuning):
    """Uses the random forest tuner because it conveniently uses the same hyperparameter that
    we are interested in here: `n_estimators`
    """

    CrossValidationTask = FearonLaitinAdaboostCV
    n_estimator_values = range(1, 51, 1)
