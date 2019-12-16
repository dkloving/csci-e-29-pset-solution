import os
from luigi import Task, Parameter, LocalTarget
from matplotlib import pyplot as plt


class RandomForestTuningBase(Task):
    """Uses cross-validation to tune the `n_estimators` parameter of a random forest

    Outputs an image with AUC plotted as a function of `n_estimators`
    """

    output_path = Parameter()

    n_estimator_values = range(5, 205, 5)

    CrossValidationTask = None  # override this for specific datasets

    NAME = "BaseTask"  # override this for specific datasets

    def requires(self):
        """Kicks off cross-validation for each value of n_estimators
        """
        for n_estimators in self.n_estimator_values:
            output_path = os.path.join(self.output().path, f"{n_estimators}_estimators")
            yield self.CrossValidationTask(
                output_path=output_path, model_params={"n_estimators": n_estimators}
            )

    def output(self):
        return LocalTarget(self.output_path)

    def run(self):
        """Collects all of the AUCs calculated during cross validation and plots them
        """
        aucs = []
        for folder in self.input():
            with open(os.path.join(folder.path, "auc.txt")) as f:
                aucs.append(float(f.read()))
        plt.figure(figsize=(12, 8))
        plt.plot(
            self.n_estimator_values, aucs, marker="o", linestyle="--", color="black"
        )
        plt.title(f"Random Forest AUC on {self.NAME}")
        plt.ylabel("AUC")
        plt.xlabel("Number of Trees")
        plt.savefig(os.path.join(self.output().path, "results.png"))

    def complete(self):
        return os.path.exists(os.path.join(self.output().path, "results.png"))
