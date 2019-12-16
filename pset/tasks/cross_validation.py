import os
from luigi import Task, IntParameter, Parameter, LocalTarget
import numpy as np
from sklearn.metrics import roc_curve, auc, roc_auc_score

from .random_forest import SambanisRandomForestCV


class CrossValidateBase(Task):
    """docstring goes here
    """
    n_folds = IntParameter(10)
    output_path = Parameter()

    ModelTask = SambanisRandomForestCV  # override this to use other datasets

    def output(self):
        return LocalTarget(self.output_path)

    def requires(self):
        """This should the folded data task
        """
        for fold in range(self.n_folds):
            output_path = os.path.join(self.output().path, str(fold))
            yield self.ModelTask(output_path=output_path, fold_id=fold)

    def run(self):
        """Accumulates all predictions and ground truths to calculate an ROC/AUC metric"""
        predictions = np.array([])
        ground_truth = np.array([])
        for folder in self.input():
            fold_predictions = np.load(os.path.join(folder.path, "predictions.npy"))
            fold_truth = np.load(os.path.join(folder.path, "ground_truth.npy"))
            predictions = np.concatenate((predictions, fold_predictions))
            ground_truth = np.concatenate((ground_truth, fold_truth))
        score = roc_auc_score(ground_truth, predictions)
        with open(os.path.join(self.output().path, "auc.txt"), 'w') as f:
            f.write(str(score))


    def complete(self):
        """This should mark the task as complete only if the expected number of files have been written.
        """
        file_count = 0
        for root, dirs, files in os.walk(self.output().path):
            file_count += len(files)
        num_files_as_expected = (file_count == self.n_folds)
        return num_files_as_expected
