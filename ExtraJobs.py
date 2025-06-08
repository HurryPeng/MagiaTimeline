from __future__ import annotations
import typing
import pytesseract
import paddleocr
import multiprocessing

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
        self.suffix: str = config["suffix"]
        self.separator: str = config["separator"]
        self.doPaddle: bool = config["doPaddle"]
        self.doTeseract: bool = config["doTesseract"]
        self.tesseractLang: str = config["tesseractLang"]

        self.filename = self.dest + self.suffix

    def apply(self, iir: IIR):
        file = open(self.filename, "w", encoding="utf-8")
        print(f"Writing to {self.filename}")
        
        paddle = paddleocr.PaddleOCR(
            text_detection_model_name="PP-OCRv4_mobile_det",
            text_detection_model_dir="./PaddleOCRModels/official_models/PP-OCRv4_mobile_det",
            text_recognition_model_name="PP-OCRv5_mobile_rec",
            text_recognition_model_dir="./PaddleOCRModels/official_models/PP-OCRv5_mobile_rec",
            use_doc_orientation_classify=False,
            use_doc_unwarping=False,
            use_textline_orientation=False,
            device="cpu",
            cpu_threads=multiprocessing.cpu_count(),
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
                recTexts = paddleResult["rec_texts"]
                paddleText: str = ""
                for line in recTexts:
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

            file.write(f"{name},{buff}\n")
            if i % 10 == 0:
                print(name)

        file.close()
