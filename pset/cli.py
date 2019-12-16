from luigi import build
from .tasks.sambanis_pipeline import SambanisRFTuning


def main():
    build([SambanisRFTuning(output_path="temp")], local_scheduler=True)
