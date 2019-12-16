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
    """This prepares the Sambanis dataset for use in k-fold cross-validation.

    It filters the features to be only those desired and then writes npz containing X and y arrays
    along with similar train / test arrays for each fold.
    """

    n_splits = IntParameter(10)
    seed = IntParameter(42)

    OUTPUT_PATH = os.path.abspath("output")
    OUTPUT_NAME = "sambanis"
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
        return LocalTarget(os.path.join(self.OUTPUT_PATH, self.OUTPUT_NAME))

    def run(self):
        with self.input().open() as f:
            data = pd.read_csv(f, usecols=self.COLS)

        X = data.drop('warstds', axis=1).values
        y = data['warstds'].values

        # save entire arrays
        os.makedirs(self.output().path, exist_ok=True)
        np.savez(os.path.join(self.output().path, "data.npz"), X=X, y=y)

        # split into folds
        cv = StratifiedKFold(n_splits=self.n_splits, shuffle=True, random_state=self.seed)

        # save train and test file for each fold
        for i, (train, test) in enumerate(cv.split(X, y)):
            k_folder = os.path.join(self.output().path, "folds", str(i))
            os.makedirs(k_folder, exist_ok=True)
            train_output_path = os.path.join(k_folder, "train.npz")
            test_output_path = os.path.join(k_folder, "test.npz")
            np.savez(train_output_path, X=X[train], y=y[train])
            np.savez(test_output_path, X=X[test], y=y[test])

    def complete(self):
        """This should mark the task as complete only if the expected number of files have been written
        however, it seems that currently luigi is incapable of recognizing the task is incomplete even
        if we always return "False".
        """
        file_count = 0
        for root, dirs, files in os.walk(self.output().path):
            file_count += len(files)
        num_files_as_expected = (file_count == self.n_splits * 2 + 1)
        return num_files_as_expected
