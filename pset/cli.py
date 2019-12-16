from luigi import build
import os
from .tasks.sambanis_pipeline import SambanisRFTuning, SambanisAdaboostTuning


def main():
    build([SambanisRFTuning(output_path=os.path.join("temp", "sambanis", "rf")),
           SambanisAdaboostTuning(output_path=os.path.join("temp", "sambanis", "adaboost"))
           ], local_scheduler=True)
