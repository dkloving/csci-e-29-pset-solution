from luigi import build
from .tasks.fold_data import SambanisSplitFolds
from .tasks.random_forest import SambanisRandomForestCV
from .tasks.cross_validation import SambanisCV
from .tasks.hyperparameter_tuning import SambanisRFTuning


def main():
    build([SambanisRFTuning(output_path='temp')]
          , local_scheduler=True)
    # build([SambanisSplitFolds()], local_scheduler=True)
