from __future__ import annotations
import pytesseract
import typing

from Util import *
from IR import *


class IIRTextExtractionPass(IIRPass):
    def __init__(self, config: dict, frameKey: type, dest: str):
        self.config: dict = config
        self.frameKey: type = frameKey
        self.dest: str = dest
        self.standaloneOutput: bool = config["standaloneOutput"]
        self.standaloneOutputSuffix: str = config["standaloneOutputSuffix"]
        self.separator: str = config["separator"]
        self.doPaddle: bool = config["doPaddle"]
        self.nonMajorBoxSuppressionMaxRatio: float = config["nonMajorBoxSuppressionMaxRatio"]
        self.nonMajorBoxSuppressionMinRank: int = config["nonMajorBoxSuppressionMinRank"]
        self.doTesseract: bool = config["doTesseract"]
        self.tesseractLang: str = config["tesseractLang"]

    def apply(self, iir: IIR):
        print(f"IIRTextExtractionPass: processing {len(iir.intervals)} intervals")
        file = None
        if self.standaloneOutput:
            filename = self.dest + self.standaloneOutputSuffix
            print(f"Standalone output enabled. Writing to {filename}")
            file = open(filename, "w", encoding="utf-8")
        else:
            print("Standalone output disabled. Writing to ass file.")

        paddle = None
        if self.doPaddle:
            suppressPaddleWarnings()
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

        for i, interval in enumerate(iir.intervals):
            buff: str = ""
            name: str = interval.getName(i)
            image: cv.Mat = interval.getAttachment(self.frameKey)

            if self.doPaddle and paddle is not None and image is not None:
                result = paddle.predict(image)
                result = result[0]
                recTexts: typing.List[str] = result["rec_texts"]
                recBoxes: np.ndarray = result["rec_boxes"]
                recPolys: typing.List[np.ndarray] = result["rec_polys"]
                recBoxSizes = [
                    (int(b[2]) - int(b[0])) * (int(b[3]) - int(b[1]))
                    for b in recBoxes
                ]
                boxSizeSum = sum(recBoxSizes)

                passesAngle: typing.List[bool] = []
                for poly in recPolys:
                    x0, y0 = poly[0]; x1, y1 = poly[1]
                    x2, y2 = poly[2]; x3, y3 = poly[3]
                    angle = (np.arctan2(y1 - y0, x1 - x0) + np.arctan2(y2 - y3, x2 - x3)) / 2
                    passesAngle.append(bool(np.abs(angle) <= np.pi / 180 * 3))

                sortedIdx = sorted(
                    range(len(recBoxSizes)),
                    key=lambda j: recBoxSizes[j] if passesAngle[j] else 0,
                    reverse=True
                )
                rankMap = [0] * len(recBoxSizes)
                for rank, origIdx in enumerate(sortedIdx):
                    rankMap[origIdx] = rank

                paddleText = ""
                for j, (line, boxSize) in enumerate(zip(recTexts, recBoxSizes)):
                    if not passesAngle[j]:
                        continue
                    if boxSize > self.nonMajorBoxSuppressionMaxRatio * boxSizeSum \
                            or rankMap[j] < self.nonMajorBoxSuppressionMinRank:
                        paddleText += line + " "
                buff += paddleText.strip()

            if self.doTesseract:
                if buff != "":
                    buff += self.separator
                if image is not None:
                    tesseractText: str = pytesseract.image_to_string(
                        ensureMat(image), config=f"-l {self.tesseractLang} --psm 6"
                    )
                    buff += tesseractText[:-1].replace("\n", "")

            if file is not None:
                file.write(f"{name},{buff}\n")
            else:
                interval.text = buff

            if i % 10 == 0:
                print(name)

        if file is not None:
            file.close()
            print(f"Output written to {file.name}")
