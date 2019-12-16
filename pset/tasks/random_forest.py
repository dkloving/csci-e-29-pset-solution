from sklearn.ensemble import RandomForestClassifier

from .fold_data import SambanisSplitFolds
from .fit_and_predict import FitPredictOnFold


class SambanisRandomForest(FitPredictOnFold):

    CLASSIFIER = RandomForestClassifier

    def requires(self):
        return SambanisSplitFolds()
