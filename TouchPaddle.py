# Touch Paddle so it downloads the required OCR models

from Strategies.BoxColourStatStrategy import *
from Strategies.DiffTextDetectionStrategy import *

if __name__ == "__main__":
    BoxColourStatStrategy.genOcrEngine()
    DiffTextDetectionStrategy.genOcrEngine()
