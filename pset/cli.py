from luigi import build
from .tasks.fold_data import SambanisSplitFolds


def main():
    build([SambanisSplitFolds()], local_scheduler=True)
