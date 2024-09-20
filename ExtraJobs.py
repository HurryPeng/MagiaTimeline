from __future__ import annotations
import typing
import pytesseract
import paddleocr

from IR import IIR
from Util import *
from AbstractFlagIndex import *
from IR import *
from Strategies.AbstractStrategy import *

class IIROcrPass(IIRPass):
    def __init__(self, config: dict, dest: str, strategy: AbstractOcrStrategy):
        self.config: dict = config
        self.dest: str = dest
        self.strategy: AbstractOcrStrategy = strategy
        self.suffix: str = config["suffix"]
        self.separator: str = config["separator"]
        self.doPaddle: bool = config["doPaddle"]
        self.paddleLang: str = config["paddleLang"]
        self.doTeseract: bool = config["doTesseract"]
        self.tesseractLang: str = config["tesseractLang"]

        self.filename = self.dest + self.suffix

    def apply(self, iir: IIR):
        file = open(self.filename, "w")
        print(f"Writing to {self.filename}")
        
        paddle = paddleocr.PaddleOCR(use_angle_cls=True, lang=self.paddleLang, show_log=False)
        ocrFrameFlagIndex: AbstractFlagIndex = self.strategy.getOcrFrameFlagIndex()

        for i, interval in enumerate(iir.intervals):
            buff: str = ""
            name: str = interval.getName(i)
            frame: av.frame.Frame = interval.getFlag(ocrFrameFlagIndex)

            img: cv.Mat = avFrame2CvMat(frame)

            if self.doPaddle:
                paddleFrame = self.strategy.cutOcrFrame(img)
                paddleResult = paddle.ocr(paddleFrame, cls=False, bin=False)
                paddleText: str = ""
                for line in paddleResult:
                    if line is None:
                        continue
                    lineText = "".join([wordInfo[1][0] for wordInfo in line])
                    paddleText += lineText + '\n'
                paddleText = paddleText.strip()
                buff += paddleText

            if self.doTeseract:
                if buff != "":
                    buff += self.separator
                tesseractFrame = self.strategy.cutCleanOcrFrame(img)
                tesseractFrame = ensureMat(tesseractFrame)
                tesseractText: str = pytesseract.image_to_string(tesseractFrame, config=f"-l {self.tesseractLang} --psm 6")
                tesseractText = tesseractText[:-1].replace("\n", "")
                buff += tesseractText

            file.write(f"{name},{buff}\n")

        file.close()
