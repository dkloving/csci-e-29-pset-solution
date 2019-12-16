import os
from luigi import Task, IntParameter, Parameter, DictParameter, LocalTarget
import numpy as np
from sklearn.metrics import roc_auc_score


class CrossValidateBase(Task):
    """docstring goes here
    """

    n_folds = IntParameter(10)
    output_path = Parameter()
    model_params = DictParameter()

    ModelTask = None  # override this to use other datasets

    def output(self):
        return LocalTarget(self.output_path)

    def requires(self):
        """Runs one ModelTask per fold
        """
        for fold in range(self.n_folds):
            output_path = os.path.join(self.output().path, str(fold))
            yield self.ModelTask(
                output_path=output_path, fold_id=fold, model_params=self.model_params
            )

    def run(self):
        """Accumulates all predictions and ground truths to calculate an ROC/AUC metric
        """
        predictions = np.array([])
        ground_truth = np.array([])
        for folder in self.input():
            fold_predictions = np.load(os.path.join(folder.path, "predictions.npy"))
            fold_truth = np.load(os.path.join(folder.path, "ground_truth.npy"))
            predictions = np.concatenate((predictions, fold_predictions))
            ground_truth = np.concatenate((ground_truth, fold_truth))
        score = roc_auc_score(ground_truth, predictions)
        with open(os.path.join(self.output().path, "auc.txt"), "w") as f:
            f.write(str(score))

    def complete(self):
        """This should mark the task as complete only if the auc file has been written.
        """
        return os.path.exists(os.path.join(self.output().path, "auc.txt"))


