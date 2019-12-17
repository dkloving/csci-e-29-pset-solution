# Final Project - Proposed Luigi Pset
This project went through several evolutions before ending up here. The original goal had been to do some massively parallel processing on AWS, managed by Luigi.
This proved to be too complicated of a task for two reasons- unfamiliarity with Docker (required for AWS), and unfamiliarity with Luigi.
While working toward a solution, I realized that this could be a useful exercise for other students as well, and so I created the Pset that I wished we had, along with one solution.

The goal of this Pset is to transform a one-off jupyter notebook project into an extendable, reusable data pipeline. This naturally involves typical data science activities,
such as cross-validation and hyperparameter tuning, but the student must think about how these activities relate in a context that extends beyond the notebook.

My solution was to create base luigi.Task classes to handle:
 - splitting data into cross-validation samples
 - fitting a model and getting test predictions from it
 - managing the cross-validation process to perform model fitting on k-folds in parallel
 - managing the hyperparameter tuning process, relying on the aforementioned cross-validation Task
 
Although I was working with fairly small datasets, this division of tasks is indended to support larger datasets and deployment to clusters. While I have not yet prepared
these tasks to run in Docker images for easy batching to a cloud compute service, doing so is the next step in this project.

To actually process my data, I was able to rely on the classes described above to create pipelines for two specific datasets. The work required to add specific datasets or specific ML
models was very modest, and can be found in `sambanis_pipeline.py` and `fearonlaitin_pipeline.py`. Adding new models is fairly easy an straightforward, but adding a new dataset takes only minutes.

My solutions to the problems below are indexed by github releases. Version 0.1.0 solves problem 1, v0.2.0 solves problem 2, and v0.3.0 solves problem 3. Later releases are to add testing and this readme. 

# PSet Description
The purpose of this pset is to help you bridge the gap between jupyter notebooks ("the old way") and reusable, extendable data processing pipelines ("the new way").
You have been provided with a notebook much like what you may find in the wild, including in your own previous data science work. Your goal is to first replicate the analysis
performed in the notebook, and then to extend that analysis by applying it to a new dataset.

## Notebook Overview
The notebook first loads some data, splits it for k-fold cross validation, does hyperparameter turning for a random forest algorithm and then also for a gradient boosting algorithm.
Take some time now to familiarize yourself with it. Your final output will be graphs like those created in the notebook.

## Problem 1
You must use Luigi to perform cross-validation and hyperparameter tuning on the provided dataset for a random forest on the `data/SambnisImp.csv` dataset.
This must be able to run in parallel, but it is up to you to decide at which level(s) that should be performed. Include comments
to explain your decision. While you are constructing your Luigi tasks, keep in mind that you will be adding another classifier and dataset later.

## Problem 2
Add the Adaboost classifier, complete with all of the cross-validation and hyperparameter tuning functionality that exists for the Random Forest classifier.
Try to reuse as much code as possible so that adding a third (or more) model will be easy. Depending on how you solved Problem 1,
you may find that you can do significant refactoring.

## Problem 3
Solve problems 1 and 2 but for the `data/fearon_laitin.csv` dataset. This should be achievable in very little time. If not, you
may be duplicating a lot of code unnecessarily and should consider how you can restructure your solution to make this easier.

## [Optional]
Add another model type or a dataset of your own choosing. Compare the time required to do this relative to doing it "the old way"
in jupyter notebooks.