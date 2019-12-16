import os
from luigi import Task, IntParameter, Parameter, LocalTarget
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from joblib import dump

from .fold_data import SambanisSplitFolds


class RandomForestCVTask(Task):
    """Fits and predicts with a RandomForest model, saving the predicted probabilities on test data

    Input files must be npz format, containing `X` and `y` arrays.

    BASE class that should be used to create a specific class for a specific dataset
    """

    output_path = Parameter()
    fold_id = IntParameter()
    n_estimators = IntParameter()

    def requires(self):
        raise NotImplementedError

    def output(self):
        return LocalTarget(self.output_path)

    def run(self):
        root_dir = os.path.join(self.input().path, "folds", str(self.fold_id))
        train_data = np.load(os.path.join(root_dir, "train.npz"))
        test_data = np.load(os.path.join(root_dir, "train.npz"))
        clf = RandomForestClassifier(n_estimators=self.n_estimators)
        clf.fit(train_data['X'], train_data['y'])
        y_pred = clf.predict_proba(test_data['X'])[:, 1]
        os.makedirs(self.output().path, exist_ok=True)
        np.save(os.path.join(self.output().path, "predictions.npy"), y_pred)
        dump(clf, os.path.join(self.output().path, "model.joblib"))

    def complete(self):
        """This should mark the task as complete only if both predictions and the pickled model files exist.
        """
        preds_exists = os.path.exists(os.path.join(self.output().path, "predictions.npy"))
        model_exists = os.path.exists(os.path.join(self.output().path, "model.joblib"))
        return preds_exists and model_exists


class SambanisRandomForestCV(RandomForestCVTask):

    def requires(self):
        return SambanisSplitFolds()
