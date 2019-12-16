import os
from luigi import Task, LocalTarget, IntParameter, Parameter

import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold


class SplitFoldsBase(Task):
    """Base class for preparing a dataset for use in k-fold cross-validation.

    It filters the features to be only those desired and then writes npz containing X and y arrays
    along with similar train / test arrays for each fold.

    Parameters that must be overridden by child class:
        OUTPUT_NAME (str): to be appended to output_path for saving data
        Y_COL (str): name of the column of the dependent variable
        FEATURE_COLS (List[str]): list of column names of independent variables
    """

    n_splits = IntParameter(10)
    seed = IntParameter(42)
    output_path = Parameter("output")

    OUTPUT_NAME = None  # set this on implementation
    Y_COL = None  # set this on implementation
    FEATURE_COLS = None  # set this on implementation

    def requires(self):
        raise NotImplementedError

    def output(self):
        return LocalTarget(os.path.join(self.output_path, self.OUTPUT_NAME))

    def run(self):
        with self.input().open() as f:
            data = pd.read_csv(f, usecols=self.FEATURE_COLS + [self.Y_COL])

        X = data.drop(self.Y_COL, axis=1).values
        y = data[self.Y_COL].values

        # save entire arrays
        os.makedirs(self.output().path, exist_ok=True)
        np.savez(os.path.join(self.output().path, "data.npz"), X=X, y=y)

        # split into folds
        cv = StratifiedKFold(
            n_splits=self.n_splits, shuffle=True, random_state=self.seed
        )

        # save train and test file for each fold
        for i, (train, test) in enumerate(cv.split(X, y)):
            k_folder = os.path.join(self.output().path, "folds", str(i))
            os.makedirs(k_folder, exist_ok=True)
            train_output_path = os.path.join(k_folder, "train.npz")
            test_output_path = os.path.join(k_folder, "test.npz")
            np.savez(train_output_path, X=X[train], y=y[train])
            np.savez(test_output_path, X=X[test], y=y[test])

    def complete(self):
        """This should mark the task as complete only if the expected number of files have been written.
        """
        file_count = 0
        for root, dirs, files in os.walk(self.output().path):
            file_count += len(files)
        num_files_as_expected = file_count == self.n_splits * 2 + 1
        return num_files_as_expected
