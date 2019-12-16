import os
from sklearn.ensemble import RandomForestClassifier
from luigi import ExternalTask, LocalTarget

from .fold_data import SplitFoldsBase
from .fit_and_predict import FitPredictOnFoldBase
from .cross_validation import CrossValidateBase
from .hyperparameter_tuning import RandomForestTuningBase


class SambanisData(ExternalTask):
    """Loads the original data via a hard-coded local path"""

    DATA_PATH = os.path.abspath("data")
    DATA_FILENAME = "SambnisImp.csv"

    def output(self):
        return LocalTarget(os.path.join(self.DATA_PATH, self.DATA_FILENAME))


class SambanisSplitFolds(SplitFoldsBase):
    """Splits the Sambanis data into k folds"""
    OUTPUT_NAME = "sambanis"
    Y_COL = "warstds"
    FEATURE_COLS = [
        "warstds",
        "ager",
        "agexp",
        "anoc",
        "army85",
        "autch98",
        "auto4",
        "autonomy",
        "avgnabo",
        "centpol3",
        "coldwar",
        "decade1",
        "decade2",
        "decade3",
        "decade4",
        "dem",
        "dem4",
        "demch98",
        "dlang",
        "drel",
        "durable",
        "ef",
        "ef2",
        "ehet",
        "elfo",
        "elfo2",
        "etdo4590",
        "expgdp",
        "exrec",
        "fedpol3",
        "fuelexp",
        "gdpgrowth",
        "geo1",
        "geo2",
        "geo34",
        "geo57",
        "geo69",
        "geo8",
        "illiteracy",
        "incumb",
        "infant",
        "inst",
        "inst3",
        "life",
        "lmtnest",
        "ln_gdpen",
        "lpopns",
        "major",
        "manuexp",
        "milper",
        "mirps0",
        "mirps1",
        "mirps2",
        "mirps3",
        "nat_war",
        "ncontig",
        "nmgdp",
        "nmdp4_alt",
        "numlang",
        "nwstate",
        "oil",
        "p4mchg",
        "parcomp",
        "parreg",
        "part",
        "partfree",
        "plural",
        "plurrel",
        "pol4",
        "pol4m",
        "pol4sq",
        "polch98",
        "polcomp",
        "popdense",
        "presi",
        "pri",
        "proxregc",
        "ptime",
        "reg",
        "regd4_alt",
        "relfrac",
        "seceduc",
        "second",
        "semipol3",
        "sip2",
        "sxpnew",
        "sxpsq",
        "tnatwar",
        "trade",
        "warhist",
        "xconst",
    ]

    def requires(self):
        return SambanisData()


class SambanisRandomForest(FitPredictOnFoldBase):
    """Random Forest classifier for Sambanis data"""

    CLASSIFIER = RandomForestClassifier

    def requires(self):
        return SambanisSplitFolds()


class SambanisRFCV(CrossValidateBase):
    ModelTask = SambanisRandomForest


class SambanisRFTuning(RandomForestTuningBase):
    CrossValidationTask = SambanisRFCV
    NAME = "Sambanis Dataset"
