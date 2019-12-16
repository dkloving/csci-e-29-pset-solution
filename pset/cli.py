from luigi import build
import os
from .tasks.sambanis_pipeline import SambanisRFTuning, SambanisAdaboostTuning
from .tasks.fearonlaitin_pipeline import FearonLaitinRFTuning, FearonLaitinAdaboostTuning

def main():
    build([SambanisRFTuning(output_path=os.path.join("temp", "sambanis", "rf")),
           SambanisAdaboostTuning(output_path=os.path.join("temp", "sambanis", "adaboost")),
           FearonLaitinRFTuning(output_path=os.path.join("temp", "fearon_laitin", "rf")),
           FearonLaitinAdaboostTuning(output_path=os.path.join("temp", "fearon_laitin", "adaboost"))
           ], local_scheduler=True)
