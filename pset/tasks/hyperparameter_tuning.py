import os
from luigi import Task, Parameter, LocalTarget
from matplotlib import pyplot as plt

from .cross_validation import SambanisCV


class RFTuningBase(Task):
    """Uses cross-validation to tune the `n_estimators` parameter of a random forest

    Outputs an image with AUC plotted as a function of `n_estimators`
    """
    output_path = Parameter()

    n_estimator_values = range(5, 205, 5)

    CrossValidationTask = None  # override this for specific datasets

    NAME = 'BaseTask'  # override this for specific datasets

    def requires(self):
        for n_estimators in self.n_estimator_values:
            output_path = os.path.join(self.output().path, f"{n_estimators}_estimators")
            yield self.CrossValidationTask(output_path=output_path, model_params={'n_estimators': n_estimators})

    def output(self):
        return LocalTarget(self.output_path)

    def run(self):
        aucs = []
        for folder in self.input():
            with open(os.path.join(folder.path, 'auc.txt')) as f:
                aucs.append(float(f.read()))
        plt.figure(figsize=(12, 8))
        plt.plot(self.n_estimator_values, aucs, marker='o', linestyle='--', color='black')
        plt.title(f'Random Forest AUC on {self.NAME}')
        plt.ylabel('AUC')
        plt.xlabel('Number of Trees')
        plt.savefig(os.path.join(self.output().path, 'results.png'))

    def complete(self):
        return os.path.exists(os.path.join(self.output().path, 'results.png'))


class SambanisRFTuning(RFTuningBase):
    CrossValidationTask = SambanisCV
    NAME = 'Sambanis Dataset'
