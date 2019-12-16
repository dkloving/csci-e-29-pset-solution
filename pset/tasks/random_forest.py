import os
from luigi import Task, IntParameter, Parameter, LocalTarget
import numpy as np
from sklearn.ensemble import RandomForestClassifier


class RandomForestEvalTask(Task):
    """Fits and predicts with a RandomForest model, saving the predicted probabilities on test data

    Input files must be npz format, containing `X` and `y` arrays.
    """

    output_path = Parameter()
    train_file = Parameter()
    test_file = Parameter()
    n_estimators = IntParameter()

    def output(self):
        return LocalTarget(self.output_path)

    def run(self):
        train_data = np.load(self.train_file)
        test_data = np.load(self.test_file)
        clf = RandomForestClassifier(n_estimators=self.n_estimators)
        clf.fit(train_data['X'], train_data['y'])
        y_pred = clf.predict_proba(test_data['X'])[:, 1]
        os.makedirs(self.output().path, exist_ok=True)
        np.save(os.path.join(self.output().path, "predictions.npy"), y_pred)

    def complete(self):
        return os.path.exists(os.path.join(self.output().path, "predictions.npy"))
