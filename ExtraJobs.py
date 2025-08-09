from __future__ import annotations
import pytesseract
import paddleocr
import typing

from IR import IIR
from Util import *
from AbstractFlagIndex import *
from IR import *
from Strategies.AbstractStrategy import *

class IIROcrPass(IIRPass):
    def __init__(self, config: dict, dest: str, strategy: AbstractExtraJobStrategy):
        self.config: dict = config
        self.dest: str = dest
        self.strategy: AbstractExtraJobStrategy = strategy
        self.standaloneOutput: bool = config["standaloneOutput"]
        self.standaloneOutputSuffix: str = config["standaloneOutputSuffix"]
        self.separator: str = config["separator"]
        self.doPaddle: bool = config["doPaddle"]
        self.nonMajorBoxSuppressionMaxRatio: float = config["nonMajorBoxSuppressionMaxRatio"]
        self.nonMajorBoxSuppressionMinRank: int = config["nonMajorBoxSuppressionMinRank"]
        self.doTeseract: bool = config["doTesseract"]
        self.tesseractLang: str = config["tesseractLang"]

    def apply(self, iir: IIR):
        file = None
        if self.standaloneOutput:
            filename = self.dest + self.standaloneOutputSuffix
            print(f"Standalone output enabled. Writing to {filename}")
            file = open(filename, "w", encoding="utf-8")
        else:
            print("Standalone output disabled. Writing to ass file.")
        
        paddle = paddleocr.PaddleOCR(
            text_detection_model_name="PP-OCRv5_mobile_det",
            text_detection_model_dir="./PaddleOCRModels/official_models/PP-OCRv5_mobile_det",
            text_recognition_model_name="PP-OCRv5_mobile_rec",
            text_recognition_model_dir="./PaddleOCRModels/official_models/PP-OCRv5_mobile_rec",
            use_doc_orientation_classify=False,
            use_doc_unwarping=False,
            use_textline_orientation=False,
            device="cpu",
            enable_mkldnn=True
        )
        extraJobFrameFlagIndex: AbstractFlagIndex = self.strategy.getExtraJobFrameFlagIndex()

        for i, interval in enumerate(iir.intervals):
            buff: str = ""
            name: str = interval.getName(i)
            img: cv.Mat = interval.getFlag(extraJobFrameFlagIndex)

            if self.doPaddle:
                paddleFrame = img
                paddleResult = paddle.predict(paddleFrame)
                paddleResult = paddleResult[0]
                recTexts: typing.List[str] = paddleResult["rec_texts"]
                recBoxes: np.ndarray = paddleResult["rec_boxes"] # List[(xmin, ymin, xmax, ymax)]
                recScores: typing.List[float] = paddleResult["rec_scores"]
                recPolys: typing.List[np.ndarray] = paddleResult["rec_polys"] # List[np.ndarray] of shape (4, 2)
                recBoxSizes = [(int(box[2]) - int(box[0])) * (int(box[3]) - int(box[1])) for box in recBoxes]
                boxSizeSum = sum(recBoxSizes)

                passesAngleTest: list[bool] = []
                for poly in recPolys:
                    x0, y0 = poly[0]
                    x1, y1 = poly[1]
                    x2, y2 = poly[2]
                    x3, y3 = poly[3]
                    angle0 = np.arctan2(y1 - y0, x1 - x0)
                    angle3 = np.arctan2(y2 - y3, x2 - x3)
                    angle = (angle0 + angle3) / 2
                    passes = np.abs(angle) <= np.pi / 180 * 10
                    passesAngleTest.append(passes)

                recBoxesSortedIndices = sorted(
                    range(len(recBoxSizes)),
                    key=lambda j: recBoxSizes[j] if passesAngleTest[j] else 0,
                    reverse=True
                )
                recBoxesRankMapping = [0] * len(recBoxSizes)
                for rank, origIdx in enumerate(recBoxesSortedIndices):
                    recBoxesRankMapping[origIdx] = rank
                recBoxesRanking = [recBoxesRankMapping[i] for i in range(len(recBoxSizes))]

                paddleText: str = ""
                for j in range(len(recTexts)):
                    line = recTexts[j]
                    box = recBoxes[j]
                    poly = recPolys[j]
                    score = recScores[j]
                    boxSize = recBoxSizes[j]
                    rank = recBoxesRanking[j]

                    if not passesAngleTest[j]:
                        continue

                    if boxSize > self.nonMajorBoxSuppressionMaxRatio * boxSizeSum or rank < self.nonMajorBoxSuppressionMinRank:
                        paddleText += line + ' '
                paddleText = paddleText.strip()
                buff += paddleText

            if self.doTeseract:
                if buff != "":
                    buff += self.separator
                tesseractFrame = img
                tesseractFrame = ensureMat(tesseractFrame)
                tesseractText: str = pytesseract.image_to_string(tesseractFrame, config=f"-l {self.tesseractLang} --psm 6")
                tesseractText = tesseractText[:-1].replace("\n", "")
                buff += tesseractText

            if file is not None:
                file.write(f"{name},{buff}\n")
            else:
                interval.text = buff

            if i % 10 == 0:
                print(name)

        if file is not None:
            file.close()
            print(f"Output written to {file.name}")
