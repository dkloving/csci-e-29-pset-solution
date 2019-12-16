import os
from luigi import Task, IntParameter, Parameter, DictParameter, LocalTarget
import numpy as np
from joblib import dump


class FitPredictOnFoldBase(Task):
    """Fits and predicts with a RandomForest model, saving the predicted probabilities on test data

    Input files must be npz format, containing `X` and `y` arrays.

    BASE class that should be used to create a specific class for a specific dataset
    """

    output_path = Parameter()
    fold_id = IntParameter()
    model_params = DictParameter({"n_estimators": 10})
    CLASSIFIER = None  # set this on implementation

    def requires(self):
        raise NotImplementedError

    def output(self):
        return LocalTarget(self.output_path)

    def run(self):
        # locate the folder containing data for this fold and read that data
        root_dir = os.path.join(self.input().path, "folds", str(self.fold_id))
        train_data = np.load(os.path.join(root_dir, "train.npz"))
        test_data = np.load(os.path.join(root_dir, "test.npz"))

        # create, fit, and inference Random Forest classifier
        clf = self.CLASSIFIER(**self.model_params)
        clf.fit(train_data["X"], train_data["y"])
        y_pred = clf.predict_proba(test_data["X"])[:, 1]

        # save the predictions, ground truth (for reference), and trained model
        os.makedirs(self.output().path, exist_ok=True)
        np.save(os.path.join(self.output().path, "predictions.npy"), y_pred)
        np.save(os.path.join(self.output().path, "ground_truth.npy"), test_data["y"])
        dump(clf, os.path.join(self.output().path, "model.joblib"))

    def complete(self):
        """This should mark the task as complete only if both predictions and the pickled model files exist.
        """
        preds_exists = os.path.exists(
            os.path.join(self.output().path, "predictions.npy")
        )
        truth_exists = os.path.exists(
            os.path.join(self.output().path, "ground_truth.npy")
        )
        model_exists = os.path.exists(os.path.join(self.output().path, "model.joblib"))
        return preds_exists and truth_exists and model_exists
