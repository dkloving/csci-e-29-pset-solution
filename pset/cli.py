from luigi import build
from .tasks.hyperparameter_tuning import SambanisRFTuning


def main():
    build([SambanisRFTuning(output_path="temp")], local_scheduler=True)
