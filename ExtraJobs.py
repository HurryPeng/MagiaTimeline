from __future__ import annotations
import pytesseract
import paddleocr
import typing

from IR import IIR
from Util import *
from AbstractFlagIndex import *
from IR import *
from Strategies.AbstractStrategy import *

class TextDetectionResult(AttachmentKey):
    """Stores text detection boxes for an image region.
    Crops are not stored; call cropTextFromImage(image) to produce them on demand.
    This class also serves as its own attachment key: use TextDetectionResult (the class
    itself) as the key when calling setAttachment/getAttachment."""
    def __init__(self, boxes: typing.Optional[typing.List[typing.Tuple[int, int, int, int]]] = None):
        self.boxes: typing.List[typing.Tuple[int, int, int, int]] = boxes if boxes is not None else []

    def cropTextFromImage(self, image: cv.Mat) -> typing.List[np.ndarray]:
        return [image[y:y + h, x:x + w].copy() for x, y, w, h in self.boxes]

class IIRTextDetectionPreprocessPass(IIRPass):
    def __init__(self, frameKey: type, config: dict):
        self.frameKey: type = frameKey
        self.nonMajorBoxSuppressionMaxRatio: float = config["nonMajorBoxSuppressionMaxRatio"]
        self.nonMajorBoxSuppressionMinRank: int = config["nonMajorBoxSuppressionMinRank"]
        suppressPaddleWarnings()
        self.detector = paddleocr.TextDetection(
            model_name="PP-OCRv4_mobile_det",
            model_dir="./PaddleOCRModels/official_models/PP-OCRv4_mobile_det",
            thresh=0.2,
            box_thresh=0.3,
            device="cpu",
            enable_mkldnn=True
        )

    def apply(self, iir: IIR):
        print(f"IIRTextDetectionPreprocessPass: processing {len(iir.intervals)} intervals")
        for i, interval in enumerate(iir.intervals):
            image: cv.Mat = interval.getAttachment(self.frameKey)
            if image is None:
                continue
            imgH, imgW = image.shape[:2]

            result = self.detector.predict(image)
            result = result[0]
            dtPolys: typing.List[np.ndarray] = result["dt_polys"]
            n = len(dtPolys)

            rawBoxes = []
            for j in range(n):
                poly = np.array(dtPolys[j], np.int32)
                x0, y0 = poly[0]
                x1, y1 = poly[1]
                x2, y2 = poly[2]
                x3, y3 = poly[3]
                angle0 = np.arctan2(y1 - y0, x1 - x0)
                angle3 = np.arctan2(y2 - y3, x2 - x3)
                angle = (angle0 + angle3) / 2
                if np.abs(angle) > np.pi / 180 * 3:
                    continue
                bx, by, bw, bh = cv.boundingRect(poly)
                bx = max(0, bx)
                by = max(0, by)
                bw = min(imgW - bx, bw)
                bh = min(imgH - by, bh)
                rawBoxes.append((bx, by, bw, bh))

            rawBoxes.sort(key=lambda b: b[2] * b[3], reverse=True)
            boxSizeSum = sum(b[2] * b[3] for b in rawBoxes)
            filteredBoxes = []
            for rank, box in enumerate(rawBoxes):
                if box[2] * box[3] <= self.nonMajorBoxSuppressionMaxRatio * boxSizeSum and rank >= self.nonMajorBoxSuppressionMinRank:
                    break
                filteredBoxes.append(box)
            filteredBoxes.sort(key=lambda b: (b[1], b[0]))  # top-to-bottom, left-to-right

            interval.setAttachment(TextDetectionResult, TextDetectionResult(filteredBoxes))

            if i % 10 == 0:
                print(interval.getName(i))


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

        recognizer = None
        if self.doPaddle:
            suppressPaddleWarnings()
            recognizer = paddleocr.TextRecognition(
                model_name="PP-OCRv5_mobile_rec",
                model_dir="./PaddleOCRModels/official_models/PP-OCRv5_mobile_rec",
                device="cpu",
                enable_mkldnn=True
            )

        frameKey: type = self.strategy.getExtraJobFrameKey()

        for i, interval in enumerate(iir.intervals):
            buff: str = ""
            name: str = interval.getName(i)
            image: cv.Mat = interval.getAttachment(frameKey)
            tdf: TextDetectionResult = interval.getAttachment(TextDetectionResult)

            if self.doPaddle and recognizer is not None:
                paddleText = ""
                if tdf is not None and image is not None and tdf.boxes:
                    for crop in tdf.cropTextFromImage(image):
                        result = recognizer.predict(crop)
                        for item in result:
                            paddleText += item["rec_text"] + " "
                paddleText = paddleText.strip()
                buff += paddleText

            if self.doTeseract:
                if buff != "":
                    buff += self.separator
                if image is not None:
                    tesseractFrame = ensureMat(image)
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
