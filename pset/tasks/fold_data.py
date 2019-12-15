import os
from luigi import Task, ExternalTask, LocalTarget, IntParameter

import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold


class SambanisData(ExternalTask):
    """Loads the original data via a hard-coded local path"""

    DATA_PATH = os.path.abspath("data")
    DATA_FILENAME = "SambnisImp.csv"

    def output(self):
        return LocalTarget(os.path.join(self.DATA_PATH, self.DATA_FILENAME))


class SambanisSplitFolds(Task):

    n_splits = IntParameter(10)
    seed = IntParameter(42)

    OUTPUT_PATH = os.path.abspath("output")
    OUTPUT_SUBFOLDER = "folds"
    COLS = ["warstds", "ager", "agexp", "anoc", "army85", "autch98", "auto4",
        "autonomy", "avgnabo", "centpol3", "coldwar", "decade1", "decade2",
        "decade3", "decade4", "dem", "dem4", "demch98", "dlang", "drel",
        "durable", "ef", "ef2", "ehet", "elfo", "elfo2", "etdo4590",
        "expgdp", "exrec", "fedpol3", "fuelexp", "gdpgrowth", "geo1", "geo2",
        "geo34", "geo57", "geo69", "geo8", "illiteracy", "incumb", "infant",
        "inst", "inst3", "life", "lmtnest", "ln_gdpen", "lpopns", "major", "manuexp", "milper",
        "mirps0", "mirps1", "mirps2", "mirps3", "nat_war", "ncontig",
        "nmgdp", "nmdp4_alt", "numlang", "nwstate", "oil", "p4mchg",
        "parcomp", "parreg", "part", "partfree", "plural", "plurrel",
        "pol4", "pol4m", "pol4sq", "polch98", "polcomp", "popdense",
        "presi", "pri", "proxregc", "ptime", "reg", "regd4_alt", "relfrac", "seceduc",
        "second", "semipol3", "sip2", "sxpnew", "sxpsq", "tnatwar", "trade",
        "warhist", "xconst"]

    def requires(self):
        return SambanisData()

    def output(self):
        return LocalTarget(os.path.join(self.OUTPUT_PATH, self.OUTPUT_SUBFOLDER))

    def run(self):
        with self.input().open() as f:
            data = pd.read_csv(f, usecols=self.COLS)

        X = data.drop('warstds', axis=1).values
        y = data['warstds'].values

        cv = StratifiedKFold(n_splits=self.n_splits, shuffle=True, random_state=self.seed)

        for i, (train, test) in enumerate(cv.split(X, y)):
            k_folder = os.path.join(self.output().path, str(i))
            os.makedirs(k_folder, exist_ok=True)
            train_output_path = os.path.join(k_folder, "train.npy")
            test_output_path = os.path.join(k_folder, "test.npy")
            np.save(train_output_path, train)
            np.save(test_output_path, test)

    def complete(self):
        path, dirs, files = next(os.walk("/usr/lib"))
        file_count = len(files)
        return file_count == self.n_splits * 2
