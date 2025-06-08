from __future__ import annotations
import pytesseract
import paddleocr

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
            text_detection_model_name="PP-OCRv4_mobile_det",
            text_detection_model_dir="./PaddleOCRModels/official_models/PP-OCRv4_mobile_det",
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

            if file is not None:
                file.write(f"{name},{buff}\n")
            else:
                interval.text = buff

            if i % 10 == 0:
                print(name)

        if file is not None:
            file.close()
            print(f"Output written to {file.name}")
