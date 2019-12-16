from luigi import build
from .tasks.fold_data import SambanisSplitFolds
from .tasks.random_forest import SambanisRandomForestCV
from .tasks.cross_validation import CrossValidateBase


def main():
    build([CrossValidateBase(output_path='temp')]
          , local_scheduler=True)
    # build([SambanisSplitFolds()], local_scheduler=True)
