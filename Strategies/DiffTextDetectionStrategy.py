from __future__ import annotations

import shutil
import typing
import enum
import collections
import dataclasses
import paddleocr
import warnings

from Util import *
from Strategies.AbstractStrategy import *
from AbstractFlagIndex import *
from Rectangle import *
from IR import *

class DiffTextDetectionStrategy(AbstractFramewiseStrategy, AbstractSpeculativeStrategy, AbstractExtraJobStrategy):
    @dataclasses.dataclass
    class TextBox:
        x: int
        y: int
        w: int
        h: int
        rawH: int

    @dataclasses.dataclass
    class DialogFeature:
        image: cv.Mat
        mask: cv.Mat
        time: str
        maxTextBoxArea: int
        maxTextBoxRawHeight: int

    DebugLogFields = [
        "time0",
        "time1",
        "merge",
        "level",
        "reason",
        "maskIou",
        "diffRate",
        "cc",
        "pcWarpDist",
        "warpDist",
        "maxWarpDist",
        "sobelIou",
        "postInpaintCommonEdgeRate",
        "removedCommonEdgeRate",
        "ocrIou",
        "isTypewriterCandidate",
        "isTypewriterRevoked",
        "oldXCoverage",
        "oldMaxTextBoxArea",
        "newMaxTextBoxArea",
        "oldMaxTextBoxRawHeight",
        "newMaxTextBoxRawHeight",
        "isSmallText",
        "oneSidedEdgeRate",
        "sobelEdgeDensity",
        "postInpaintUnionEdgeRate",
        "removedUnionEdgeRate",
        "commonEdgeBaseArea",
        "unionEdgeBaseArea",
        "commonUnionRemovalGap",
        "postInpaintProbValue",
    ]

    class FlagIndex(AbstractFlagIndex):
        Dialog = enum.auto()
        DialogVal = enum.auto()
        DialogFeat = enum.auto()
        DialogFeatJump = enum.auto()

        @classmethod
        def getDefaultFlagsImpl(cls) -> typing.List[typing.Any]:
            return [
                False,
                0.0,
                None,
                False,
            ]
        
    @staticmethod
    def genTextDetectionEngine() -> paddleocr.TextDetection:
        warnings.filterwarnings("ignore", 
                        message="No ccache found", 
                        category=UserWarning,
                        module="paddle.utils.cpp_extension")
        return paddleocr.TextDetection(
            model_name="PP-OCRv4_mobile_det",
            model_dir="./PaddleOCRModels/official_models/PP-OCRv4_mobile_det",
            thresh=0.2,
            box_thresh=0.3,
            device="cpu"
        )

    def __init__(self, config: dict, contentRect: AbstractRectangle) -> None:
        AbstractStrategy.__init__(self, contentRect)
        AbstractSpeculativeStrategy.__init__(self)

        self.textDetectionEngine = DiffTextDetectionStrategy.genTextDetectionEngine()
        self.textDetectionAdapter = PaddleTextDetectionAdapter(self.textDetectionEngine)

        self.rectangles: collections.OrderedDict[str, AbstractRectangle] = collections.OrderedDict()
        self.rectangles["dialogRect"] = RatioRectangle(contentRect, *config["dialogRect"])

        self.featureThreshold: float = config["featureThreshold"]
        self.boxExpansion: float = config["boxExpansion"]
        self.smallTextBoxAreaThreshold: int = 60000
        self.nonMajorBoxSuppressionMaxRatio: float = config["nonMajorBoxSuppressionMaxRatio"]
        self.colourTolerance: int = config["colourTolerance"]
        self.minMaskIou: float = 0.5
        self.minOcrIou: float = 0.10
        self.commonEdgeRemovalMergeThreshold: float = 0.70
        self.commonEdgeRemovalMergeMinSobelIou: float = 0.70
        self.commonEdgeRemovalSplitThreshold: float = 0.25
        self.commonEdgeRemovalMergeMaxOneSidedEdgeRate: float = 0.30
        self.commonEdgeRemovalMergeMinUnionEdgeRemovalRate: float = 0.80
        self.maxWarpTextBoxMinSideRatio: float = 0.60
        self.sobelShortcutMaxEdgeDensity: float = 0.45
        self.probMapThreshold: float = 0.5
        self.iirPassDenoiseMinTime: int = config["iirPassDenoiseMinTime"]
        self.enableShortCircuit: bool = config["enableShortCircuit"]
        self.debugLevel: int = config["debugLevel"]

        self.enableTypewriter: bool = config["enableTypewriter"]
        self.typewriterXCoverageThreshold: float = config["typewriterXCoverageThreshold"]
        self.typewriterAreaRatioThreshold: float = config["typewriterAreaRatioThreshold"]

        self.dialogRect = self.rectangles["dialogRect"]

        self.cvPasses = [self.cvPassDialog]

        self.fpirPasses = collections.OrderedDict()

        self.fpirToIirPasses = collections.OrderedDict()
        self.fpirToIirPasses["fpirPassBuildIntervals"] = FPIRPassBooleanBuildIntervals(
            DiffTextDetectionStrategy.FlagIndex.Dialog
        )

        self.iirPasses = collections.OrderedDict()
        
        self.specIirPasses = collections.OrderedDict()
        self.specIirPasses["iirPassMerge"] = IIRPassMerge(
            lambda iir, interval0, interval1:
                iir.ms2Timestamp(self.iirPassDenoiseMinTime) > interval0.dist(interval1) and
                self.decideFeatureMerge(
                    [interval0.getAttachment(AbstractSpeculativeStrategy.AggregatedFeatureKey)],
                    [interval1.getAttachment(AbstractSpeculativeStrategy.AggregatedFeatureKey)]
                ),
            debug=self.debugLevel > 0
        )
        self.specIirPasses["iirPassDenoise"] = IIRPassDenoise(DiffTextDetectionStrategy.FlagIndex.Dialog.name, self.iirPassDenoiseMinTime)
        self.specIirPasses["iirPassMerge2"] = self.specIirPasses["iirPassMerge"]

        self.statDecideFeatureMerge = 0
        self.statDecideFeatureMergeDiff = 0
        self.statDecideFeatureMergeComputeECC = 0
        self.statDecideFeatureMergeFindTransformECC = 0
        self.statDecideFeatureMergeInpaint = 0
        self.statDecideFeatureMergeOCR = 0
        
        if self.debugLevel == 1:
            self.log = open("dtdLog.csv", "w", encoding="utf-8")
            self.log.write(",".join(self.DebugLogFields) + "\n")
            self.log.flush()
            if os.path.exists("./dtdDebug"):
                # Remove whole dir and all contents
                shutil.rmtree("./dtdDebug")
            os.makedirs("./dtdDebug", exist_ok=True)

    @classmethod
    def getFlagIndexType(cls) -> typing.Type[AbstractFlagIndex]:
        return cls.FlagIndex
    
    @staticmethod
    def isEmptyFeature(feature: DialogFeature | None) -> bool:
        return feature is None

    @staticmethod
    def xProjectionCoverage(mask: cv.Mat) -> int:
        return int(np.count_nonzero(np.any(mask > 0, axis=0)))

    @staticmethod
    def yProjectionCoverage(mask: cv.Mat) -> int:
        return int(np.count_nonzero(np.any(mask > 0, axis=1)))

    def getRectangles(self) -> collections.OrderedDict[str, AbstractRectangle]:
        return self.rectangles

    def getCvPasses(self) -> typing.List[typing.Callable[[cv.Mat, FramePoint], bool]]:
        return self.cvPasses

    def getFpirPasses(self) -> collections.OrderedDict[str, FPIRPass]:
        return self.fpirPasses

    def getFpirToIirPasses(self) -> collections.OrderedDict[str, FPIRPassBuildIntervals]:
        return self.fpirToIirPasses

    def getIirPasses(self) -> collections.OrderedDict[str, IIRPass]:
        return self.iirPasses
    
    def getSpecIirPasses(self) -> collections.OrderedDict[str, IIRPass]:
        return self.specIirPasses

    def getDebugString(self) -> str:
        if self.debugLevel < 1:
            return ""
        return (
            f"statDecideFeatureMerge {self.statDecideFeatureMerge}\n"
            f"statDecideFeatureMergeDiff {self.statDecideFeatureMergeDiff}\n"
            f"statDecideFeatureMergeComputeECC {self.statDecideFeatureMergeComputeECC}\n"
            f"statDecideFeatureMergeFindTransformECC {self.statDecideFeatureMergeFindTransformECC}\n"
            f"statDecideFeatureMergeInpaint {self.statDecideFeatureMergeInpaint}\n"
            f"statDecideFeatureMergeOCR {self.statDecideFeatureMergeOCR}\n"
        )
    
    def decideFeatureMerge(self, oldFeatures: typing.List[typing.Any], newFeatures: typing.List[typing.Any]) -> bool:
        self.statDecideFeatureMerge += 1

        oldFeature: DiffTextDetectionStrategy.DialogFeature = oldFeatures[0]
        newFeature: DiffTextDetectionStrategy.DialogFeature = newFeatures[0]

        if self.isEmptyFeature(oldFeature) and self.isEmptyFeature(newFeature):
            return True
        if self.isEmptyFeature(oldFeature) or self.isEmptyFeature(newFeature):
            return False

        oldImage = oldFeature.image
        oldMask = oldFeature.mask
        oldTimeStr = oldFeature.time
        oldMaxTextBoxArea = oldFeature.maxTextBoxArea
        oldMaxTextBoxRawHeight = oldFeature.maxTextBoxRawHeight
        newImage = newFeature.image
        newMask = newFeature.mask
        newTimeStr = newFeature.time
        newMaxTextBoxArea = newFeature.maxTextBoxArea
        newMaxTextBoxRawHeight = newFeature.maxTextBoxRawHeight
        isSmallText = (
            oldMaxTextBoxArea > 0 and
            newMaxTextBoxArea > 0 and
            max(oldMaxTextBoxArea, newMaxTextBoxArea) <= self.smallTextBoxAreaThreshold
        )
        maskIou = 0.0
        diffRate = 0.0
        cc = 0.0
        pcWarpDist = 0.0
        warpDist = 0.0
        sobelIou = 0.0
        postInpaintCommonEdgeRate = 0.0
        removedCommonEdgeRate = 0.0
        isTypewriterCandidate = False
        isTypewriterRevoked = False
        oldXCoverage = 0.0
        oneSidedEdgeRate = 0.0
        sobelEdgeDensity = 0.0
        postInpaintUnionEdgeRate = 0.0
        removedUnionEdgeRate = 0.0
        commonUnionRemovalGap = 0.0
        commonEdgeBaseArea = 0.0
        unionEdgeBaseArea = 0.0
        maxWarpDist = 0.0
        postInpaintProbValue = 0.0

        if self.debugLevel == 1:
            # Save oldImage and newImage to "dtdDebug/<oldTimeStr>.png" and "dtdDebug/<newTimeStr>.png"
            oldTimeStrSemicolon = oldTimeStr.replace(":", ";")
            newTimeStrSemicolon = newTimeStr.replace(":", ";")
            # Remove dir and rebuild
            def saveFrames():
                imwriteAsync(f"./dtdDebug/{oldTimeStrSemicolon}-{newTimeStrSemicolon}-0old.png", oldImage)
                imwriteAsync(f"./dtdDebug/{oldTimeStrSemicolon}-{newTimeStrSemicolon}-1new.png", newImage)
            def saveExtra(step: str, image: cv.Mat):
                imwriteAsync(f"./dtdDebug/{oldTimeStrSemicolon}-{newTimeStrSemicolon}-{step}.png", image)

        def writeDebugDecision(
            merge: bool,
            level: int,
            reason: str,
            ocrIou: float = 0.0,
            images: typing.Sequence[typing.Tuple[str, cv.Mat]] = (),
        ) -> None:
            if self.debugLevel != 1:
                return
            values = [
                oldTimeStr,
                newTimeStr,
                merge,
                level,
                reason,
                maskIou,
                diffRate,
                cc,
                pcWarpDist,
                warpDist,
                maxWarpDist,
                sobelIou,
                postInpaintCommonEdgeRate,
                removedCommonEdgeRate,
                ocrIou,
                isTypewriterCandidate,
                isTypewriterRevoked,
                oldXCoverage,
                oldMaxTextBoxArea,
                newMaxTextBoxArea,
                oldMaxTextBoxRawHeight,
                newMaxTextBoxRawHeight,
                isSmallText,
                oneSidedEdgeRate,
                sobelEdgeDensity,
                postInpaintUnionEdgeRate,
                removedUnionEdgeRate,
                commonEdgeBaseArea,
                unionEdgeBaseArea,
                commonUnionRemovalGap,
                postInpaintProbValue,
            ]
            self.log.write(",".join(str(value) for value in values) + "\n")
            saveFrames()
            for step, image in images:
                saveExtra(step, image)

        # Quick mask iou check before performing ocr on the intersection of the two images
        intersectMask = cv.bitwise_and(oldMask, newMask)
        unionMask = cv.bitwise_or(oldMask, newMask)
        intersectArea = np.sum(intersectMask) / 255
        unionArea = np.sum(unionMask) / 255
        
        if unionArea == 0:
            writeDebugDecision(False, 0, "empty mask")
            return False
        
        maskIou = intersectArea / unionArea

        if self.enableTypewriter:
            oldArea = np.sum(oldMask) / 255
            newArea = np.sum(newMask) / 255
            oldXWidth = self.xProjectionCoverage(oldMask)
            intersectXWidth = self.xProjectionCoverage(intersectMask)
            oldXCoverage = intersectXWidth / oldXWidth if oldXWidth > 0 else 0.0
            areaRatio = newArea / oldArea if oldArea > 0 else 0.0
            if oldXCoverage >= self.typewriterXCoverageThreshold and areaRatio > self.typewriterAreaRatioThreshold:
                isTypewriterCandidate = True

        self.statDecideFeatureMergeDiff += 1
        
        diffMask: cv.Mat = rgbDiffMask(oldImage, newImage, self.colourTolerance)
        # The rate of pixels that are close enough
        diffArea = np.sum(diffMask) / 255
        diffRate = diffArea / unionArea
        if self.enableShortCircuit:
            if diffRate < 0.05:
                writeDebugDecision(True, 1, "diff rate too low")
                return True

            if maskIou < self.minMaskIou and not isTypewriterCandidate:
                writeDebugDecision(False, 1, "mask iou too low", images=[
                    ("2oldMask", oldMask),
                    ("3newMask", newMask),
                ])
                return False
        
        # Pre-ECC Sobel
        oldImageSobel = rgbSobel(oldImage, 1)
        newImageSobel = rgbSobel(newImage, 1)
        
        # ECC on Sobel
        self.statDecideFeatureMergeComputeECC += 1

        warp = np.eye(2, 3, dtype=np.float32)
        ccInit: float = cv.computeECC(
            templateImage=newImageSobel,
            inputImage=oldImageSobel,
            inputMask=unionMask,
        )
        cc = ccInit

        oldImageSobelMasked = cv.bitwise_and(oldImageSobel, oldImageSobel, mask=oldMask)
        newImageSobelMasked = cv.bitwise_and(newImageSobel, newImageSobel, mask=newMask)
        oldImageSobelMaskedF32 = np.float32(oldImageSobelMasked)
        newImageSobelMaskedF32 = np.float32(newImageSobelMasked)
        oldTextMinSide = min(self.xProjectionCoverage(oldMask), self.yProjectionCoverage(oldMask))
        newTextMinSide = min(self.xProjectionCoverage(newMask), self.yProjectionCoverage(newMask))
        rawTextMinSide = min(oldTextMinSide, newTextMinSide) / (1.0 + 2.0 * self.boxExpansion)
        maxWarpDist = min(50, self.maxWarpTextBoxMinSideRatio * rawTextMinSide)
        (shiftX, shiftY), response = phaseCorrelateMaxRes(
            src1=newImageSobelMaskedF32,
            src2=oldImageSobelMaskedF32,
            maxResolution=1800
        )
        if response > 0.1:
            warp = np.array([[1, 0, shiftX], [0, 1, shiftY]], dtype=np.float32)
        pcWarpDist = np.linalg.norm(warp[0:2, 2])
        if pcWarpDist > 1.5 and pcWarpDist < maxWarpDist and cc < 0.99:
            try:
                self.statDecideFeatureMergeFindTransformECC += 1
                cc, warp = cv.findTransformECC(
                    templateImage=newImageSobel,
                    inputImage=oldImageSobel,
                    warpMatrix=warp,
                    motionType=cv.MOTION_TRANSLATION,
                    criteria=(cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 10, 0.01),
                    inputMask=unionMask,
                )
            except cv.error:
                cc = 0
                warp = np.eye(2, 3, dtype=np.float32)

        warpedImage = newImage
        warpDist = np.linalg.norm(warp[0:2, 2])
        if warpDist > 1 and warpDist < maxWarpDist and cc > ccInit:
            warpedImage = cv.warpAffine(newImage, warp, (newImage.shape[1], newImage.shape[0]), flags=cv.INTER_LINEAR)
            warpedMask = cv.warpAffine(unionMask, warp, (newImage.shape[1], newImage.shape[0]), flags=cv.INTER_LINEAR)
            intersectMask = cv.bitwise_and(oldMask, warpedMask)
            unionMask = cv.bitwise_or(oldMask, warpedMask)
            if isTypewriterCandidate:
                intersectXWidth = self.xProjectionCoverage(intersectMask)
                oldXCoverage = intersectXWidth / oldXWidth if oldXWidth > 0 else 0.0
                if oldXCoverage < self.typewriterXCoverageThreshold:
                    isTypewriterCandidate = False
                    isTypewriterRevoked = True

        if isTypewriterCandidate:
            czNonZero = cv.findNonZero(intersectMask)
            if czNonZero is not None:
                typewriterTextHeights = [
                    height for height in [oldMaxTextBoxRawHeight, newMaxTextBoxRawHeight]
                    if height > 0
                ]
                typewriterTextHeight = min(typewriterTextHeights) if typewriterTextHeights else cv.boundingRect(czNonZero)[3]
                erosionRadius = max(1, int(typewriterTextHeight * self.boxExpansion))
                erosionRadius = int(erosionRadius * 1.5) # erode even more in order not to leak strokes in
                k = max(3, 2 * erosionRadius + 1)
                iouMask = cv.morphologyEx(intersectMask, cv.MORPH_ERODE, cv.getStructuringElement(cv.MORPH_ELLIPSE, (k, k)))
            else:
                iouMask = intersectMask
            iouArea = np.sum(iouMask) / 255
        else:
            iouMask = unionMask
            iouArea = np.sum(iouMask) / 255

        # Sobel Iou Filtering

        oldImageSobel = rgbSobel(oldImage, 1)
        warpedImageSobel = rgbSobel(warpedImage, 1)
        oldImageSobelBin = cv.threshold(oldImageSobel, 32, 255, cv.THRESH_BINARY)[1]
        warpedImageSobelBin = cv.threshold(warpedImageSobel, 32, 255, cv.THRESH_BINARY)[1]

        oldImageSobelBinDilate = cv.morphologyEx(oldImageSobelBin, cv.MORPH_DILATE, cv.getStructuringElement(cv.MORPH_ELLIPSE, (3, 3)))
        warpedImageSobelBinDilate = cv.morphologyEx(warpedImageSobelBin, cv.MORPH_DILATE, cv.getStructuringElement(cv.MORPH_ELLIPSE, (3, 3)))
        unionSobelBin = cv.bitwise_or(oldImageSobelBin, warpedImageSobelBin) # No dilate for union
        unionSobelBinMasked = cv.bitwise_and(unionSobelBin, unionMask)
        intersectSobelBin = cv.bitwise_and(oldImageSobelBinDilate, warpedImageSobelBinDilate)
        intersectSobelBinMasked = cv.bitwise_and(intersectSobelBin, unionSobelBinMasked)
        intersectSobelBinMaskedDilate = cv.morphologyEx(intersectSobelBinMasked, cv.MORPH_DILATE, cv.getStructuringElement(cv.MORPH_ELLIPSE, (3, 3)))
        iouUnionSobelBinMasked = cv.bitwise_and(unionSobelBin, iouMask)
        oldOnlySobelBin = cv.bitwise_and(oldImageSobelBin, cv.bitwise_not(warpedImageSobelBinDilate))
        warpedOnlySobelBin = cv.bitwise_and(warpedImageSobelBin, cv.bitwise_not(oldImageSobelBinDilate))
        oneSidedSobelBin = cv.bitwise_or(oldOnlySobelBin, warpedOnlySobelBin)
        oneSidedSobelBinMasked = cv.bitwise_and(oneSidedSobelBin, iouMask)
        commonEdgeBaseMask = cv.bitwise_and(intersectSobelBin, iouUnionSobelBinMasked)
        unionEdgeBaseArea = np.sum(iouUnionSobelBinMasked) + 1e-6
        commonEdgeBaseArea = np.sum(commonEdgeBaseMask)
        commonEdgeBaseAreaSafe = commonEdgeBaseArea + 1e-6
        sobelIou = commonEdgeBaseArea / unionEdgeBaseArea
        sobelEdgeDensity = (np.sum(iouUnionSobelBinMasked) / 255) / (iouArea + 1e-6)
        oneSidedEdgeRate = (np.sum(oneSidedSobelBinMasked) / 255) / ((np.sum(iouUnionSobelBinMasked) / 255) + 1e-6)

        if self.enableShortCircuit:
            if sobelIou > 0.9 and sobelEdgeDensity < self.sobelShortcutMaxEdgeDensity:
                writeDebugDecision(True, 3, "sobel iou too high", images=[
                    ("2oldSobel", oldImageSobelBin),
                    ("3newSobel", warpedImageSobelBin),
                    ("4unionSobel", unionSobelBinMasked),
                    ("5intersectSobel", intersectSobelBinMasked),
                    ("6iouMask", iouMask),
                ])
                return True

        # Inpainting

        self.statDecideFeatureMergeInpaint += 1

        diffMask = rgbDiffMask(oldImage, warpedImage, self.colourTolerance)

        inpaintMask = cv.bitwise_not(diffMask)

        # Allow edge only when it is a common sobel
        inpaintMaskEdge = cv.morphologyEx(inpaintMask, cv.MORPH_GRADIENT, cv.getStructuringElement(cv.MORPH_ELLIPSE, (3, 3)))
        cv.copyTo(src=intersectSobelBinMaskedDilate, dst=inpaintMask, mask=inpaintMaskEdge)

        # Allow union sobel area only when it is also a common sobel
        inpaintMaskAndIntersectSobel = cv.bitwise_and(inpaintMask, intersectSobelBinMasked)
        cv.copyTo(src=inpaintMaskAndIntersectSobel, dst=inpaintMask, mask=unionSobelBin)

        if not isSmallText:
            # Denoise small black dots. For small thin text this can erase the strokes we need OCR to see.
            inpaintMaskDilate = cv.morphologyEx(inpaintMask, cv.MORPH_DILATE, cv.getStructuringElement(cv.MORPH_ELLIPSE, (3, 3)))
            inpaintMaskDilateErode = cv.morphologyEx(inpaintMaskDilate, cv.MORPH_ERODE, cv.getStructuringElement(cv.MORPH_ELLIPSE, (5, 5)))
            inpaintMask = cv.bitwise_or(inpaintMask, inpaintMaskDilateErode)

        inpaintMask = cv.bitwise_and(inpaintMask, unionMask)

        inpaintBaseImage = warpedImage
        if isSmallText:
            useOldPixel = (oldImageSobel >= warpedImageSobel)[:, :, np.newaxis]
            inpaintBaseImage = np.where(useOldPixel, oldImage, warpedImage).astype(np.uint8)
        
        # Blur the edges around the inpaint area, using pixels not in the inpaint area
        warpedImageNoInpaintMask = cv.bitwise_and(inpaintBaseImage, inpaintBaseImage, mask=cv.bitwise_not(inpaintMask))
        warpedImageNoInpaintMaskBlur = cv.stackBlur(warpedImageNoInpaintMask, (21, 21))
        warpedImageNoInpaintMaskDenom = cv.stackBlur(cv.bitwise_not(inpaintMask), (21, 21))
        warpedImageNoInpaintMaskDenom[warpedImageNoInpaintMaskDenom == 0] = 1
        warpedImageNoInpaintMaskDenom = cv.merge([warpedImageNoInpaintMaskDenom] * 3)
        inpaintBase = cv.divide(warpedImageNoInpaintMaskBlur, warpedImageNoInpaintMaskDenom, scale=256, dtype=cv.CV_8U)

        inpaintIntermediate = cv.inpaint(inpaintBase, inpaintMask, 1, cv.INPAINT_TELEA)

        warpedImageInpaint = inpaintBaseImage.copy()
        cv.copyTo(src=inpaintIntermediate, dst=warpedImageInpaint, mask=inpaintMask)

        # Reduce sharpeness inside inpainted area
        warpedImageInpaintBlur = cv.stackBlur(warpedImageInpaint, (11, 11))
        cv.copyTo(src=warpedImageInpaintBlur, dst=warpedImageInpaint, mask=inpaintMask)

        # Post-inpaint common edge removal filtering

        warpedImageInpaintSobel = rgbSobel(warpedImageInpaint, 1)
        warpedImageInpaintSobelBin = cv.threshold(warpedImageInpaintSobel, 32, 255, cv.THRESH_BINARY)[1]
        warpedImageInpaintSobelBinDilate = cv.morphologyEx(warpedImageInpaintSobelBin, cv.MORPH_DILATE, cv.getStructuringElement(cv.MORPH_ELLIPSE, (5, 5)))
        intersectPostInpaintSobel = cv.bitwise_and(oldImageSobelBinDilate, warpedImageInpaintSobelBinDilate)
        remainingCommonEdgeMask = cv.bitwise_and(intersectPostInpaintSobel, commonEdgeBaseMask)
        remainingUnionEdgeMask = cv.bitwise_and(warpedImageInpaintSobelBinDilate, iouUnionSobelBinMasked)
        if commonEdgeBaseArea > 0:
            postInpaintCommonEdgeRate = np.sum(remainingCommonEdgeMask) / commonEdgeBaseAreaSafe
            removedCommonEdgeRate = 1.0 - postInpaintCommonEdgeRate
        else:
            postInpaintCommonEdgeRate = 0.0
            removedCommonEdgeRate = 0.0
        if unionEdgeBaseArea > 0:
            postInpaintUnionEdgeRate = np.sum(remainingUnionEdgeMask) / unionEdgeBaseArea
            removedUnionEdgeRate = 1.0 - postInpaintUnionEdgeRate
        else:
            postInpaintUnionEdgeRate = 0.0
            removedUnionEdgeRate = 0.0
        commonUnionRemovalGap = removedCommonEdgeRate - removedUnionEdgeRate

        if self.enableShortCircuit:
            if commonEdgeBaseArea > 0 and removedCommonEdgeRate < self.commonEdgeRemovalSplitThreshold:
                writeDebugDecision(False, 3, "common edge removal too low", images=[
                    ("2oldSobel", oldImageSobelBin),
                    ("3newSobel", warpedImageSobelBin),
                    ("4unionSobel", unionSobelBinMasked),
                    ("5intersectSobel", intersectSobelBinMasked),
                    ("6diffMask", diffMask),
                    ("7inpaintMask", inpaintMask),
                    ("8inpaint", warpedImageInpaint),
                    ("10iouMask", iouMask),
                ])
                return False
            if (
                removedCommonEdgeRate > self.commonEdgeRemovalMergeThreshold and
                sobelIou > self.commonEdgeRemovalMergeMinSobelIou and
                oneSidedEdgeRate < self.commonEdgeRemovalMergeMaxOneSidedEdgeRate and
                removedUnionEdgeRate > self.commonEdgeRemovalMergeMinUnionEdgeRemovalRate
            ):
                writeDebugDecision(True, 3, "common edge removal too high", images=[
                    ("2oldSobel", oldImageSobelBin),
                    ("3newSobel", warpedImageSobelBin),
                    ("4unionSobel", unionSobelBinMasked),
                    ("5intersectSobel", intersectSobelBinMasked),
                    ("6diffMask", diffMask),
                    ("7inpaintMask", inpaintMask),
                    ("8inpaint", warpedImageInpaint),
                    ("10iouMask", iouMask),
                ])
                return True

        # Stage 7: Post-inpaint text detection
        inpaintBoxes, inpaintProb = self.detectTextBoxesAndProbMap(warpedImageInpaint)
        inpaintProbInt8 = np.clip(inpaintProb * 255, 0, 255).astype(np.uint8)

        # Build OCR mask from adapter boxes (same logic as ocrPass/detectTextBoxes)
        self.statDecideFeatureMergeOCR += 1
        inpaintOcrMask = np.zeros_like(iouMask)
        for box in inpaintBoxes:
            inpaintOcrMask[box.y:box.y+box.h, box.x:box.x+box.w] = 255

        ocrIntersectMask = cv.bitwise_and(inpaintOcrMask, iouMask)
        ocrIntersectVal = np.mean(ocrIntersectMask)
        ocrIntersectArea = np.sum(ocrIntersectMask) / 255
        ocrIou = ocrIntersectArea / iouArea if iouArea > 0 else 0.0
        ocrDecision: bool = ocrIntersectVal < self.featureThreshold or ocrIou < self.minOcrIou

        # Compute prob metrics for dtdLog
        postInpaintProbValue = 0.0

        iouMaskF = iouMask.astype(np.float32) / 255.0
        iouAreaF = np.sum(iouMaskF)
        if iouAreaF > 0:
            postInpaintProbValue = float(
                np.sum((inpaintProb > self.probMapThreshold).astype(np.float32) * iouMaskF) / iouAreaF
            )

        # Heatmap split override: probHighRate > 0.05 means significant text signal
        # in iouMask region. Override OCR's merge to split when heatmap sees text.
        heatmapSplitOverride = False
        if postInpaintProbValue > 0.05:
            if ocrDecision:  # OCR says merge, but heatmap sees text
                heatmapSplitOverride = True

        # Decision priority: heatmap override > OCR
        if heatmapSplitOverride:
            finalDecision = False  # split (correct the false merge)
            decisionReason = f"heatmap split override (highRate={postInpaintProbValue:.4f})"
        else:
            finalDecision = ocrDecision
            decisionReason = "ocr decision"

        writeDebugDecision(finalDecision, 4, decisionReason, ocrIou=ocrIou, images=[
            ("2oldSobel", oldImageSobelBin),
            ("3newSobel", warpedImageSobelBin),
            ("4unionSobel", unionSobelBinMasked),
            ("5intersectSobel", intersectSobelBinMasked),
            ("6diffMask", diffMask),
            ("7inpaintMask", inpaintMask),
            ("8inpaint", warpedImageInpaint),
            ("9ocrMask", inpaintOcrMask),
            ("10iouMask", iouMask),
            ("11inpaintProb", inpaintProbInt8),
        ])
        return finalDecision
    
    def aggregateFeatures(self, features: typing.List[typing.Any]) -> typing.Any:
        # Return the last feature
        # For typewriter animation, the last frame is the one that contains the most test 
        return features[-1]

    def isFpNonEmpty(self, fp: FramePoint) -> bool:
        return fp.getFlag(DiffTextDetectionStrategy.FlagIndex.Dialog)

    def getFpFeature(self, fp: FramePoint) -> typing.Any:
        return fp.getFlag(DiffTextDetectionStrategy.FlagIndex.DialogFeat)

    def freeFpFeature(self, fp: FramePoint) -> None:
        fp.setFlag(DiffTextDetectionStrategy.FlagIndex.DialogFeat, None)

    def cutExtraJobFrame(self, frame: cv.Mat) -> cv.Mat:
        return self.dialogRect.cutRoi(frame)

def detectTextBoxesAndProbMap(self, frame: cv.Mat) -> typing.Tuple[typing.List[DiffTextDetectionStrategy.TextBox], np.ndarray]:
        """Run adapter.detect() with scale-down, returning (boxes, probMap).
        Boxes are TextBox(x, y, w, h, rawH) in original frame coords.
        Prob map is resized to original frame size.
        """
        imgH, imgW = frame.shape[:2]
        scaleDown = 1
        while imgH // scaleDown > 960 or imgW // scaleDown > 960:
            scaleDown *= 2

        detectFrame = frame
        if scaleDown > 1:
            detectFrame = cv.resize(frame, (imgW // scaleDown, imgH // scaleDown),
                                    interpolation=cv.INTER_LINEAR)

        result = self.textDetectionAdapter.detect(detectFrame)

        # Resize prob map back to original frame size
        if scaleDown > 1:
            probMap = cv.resize(result.probabilityMap, (imgW, imgH), interpolation=cv.INTER_LINEAR)
        else:
            probMap = result.probabilityMap

        boxes = []
        for i in range(len(result.boxes)):
            wordInfo = np.array(result.boxes[i], np.int32).reshape(-1, 2)
            if len(wordInfo) < 4:
                continue
            if scaleDown > 1:
                wordInfo = wordInfo * scaleDown
            x0, y0 = wordInfo[0]
            x1, y1 = wordInfo[1]
            x2, y2 = wordInfo[2]
            x3, y3 = wordInfo[3]
            angle0 = np.arctan2(y1 - y0, x1 - x0)
            angle3 = np.arctan2(y2 - y3, x2 - x3)
            angle = (angle0 + angle3) / 2
            if np.abs(angle) > np.pi / 180 * 3:
                continue
            bx, by, bw, bh = cv.boundingRect(wordInfo)
            rawH = bh
            expand = int(bh * self.boxExpansion)
            bx = max(0, bx - expand)
            by = max(0, by - expand)
            bw = max(0, min(imgW - bx, bw + 2 * expand))
            bh = max(0, min(imgH - by, bh + 2 * expand))
            boxes.append(DiffTextDetectionStrategy.TextBox(x=bx, y=by, w=bw, h=bh, rawH=rawH))

        return boxes, probMap

    def cvPassDialog(self, frame: cv.Mat, framePoint: FramePoint) -> bool:

        image = self.cutExtraJobFrame(frame)

        mask, dialogVal, debugFrame, maxTextBoxArea, maxTextBoxRawHeight = self.ocrPass(image)

        hasDialog = dialogVal > self.featureThreshold

        feat: DiffTextDetectionStrategy.DialogFeature | None = None
        if hasDialog:
            timeString = framePoint.timeString()
            feat = DiffTextDetectionStrategy.DialogFeature(
                image=image,
                mask=mask,
                time=timeString,
                maxTextBoxArea=maxTextBoxArea,
                maxTextBoxRawHeight=maxTextBoxRawHeight,
            )

        framePoint.setFlag(DiffTextDetectionStrategy.FlagIndex.Dialog, hasDialog)
        framePoint.setFlag(DiffTextDetectionStrategy.FlagIndex.DialogVal, dialogVal)
        framePoint.setFlag(DiffTextDetectionStrategy.FlagIndex.DialogFeat, feat)

        return False

    def ocrPass(self, frame: cv.Mat) -> typing.Tuple[cv.Mat, float, cv.Mat | None, int, int]:
        # returns mask, dialogVal, debugFrame, maxTextBoxArea, maxTextBoxRawHeight

        boxes, _ = self.detectTextBoxesAndProbMap(frame)

        mask: cv.Mat = np.zeros_like(frame[:, :, 0])
        boxes = sorted(boxes, key=lambda box: box.w * box.h, reverse=True)
        maxTextBoxArea = boxes[0].w * boxes[0].h if boxes else 0
        maxTextBoxRawHeight = 0
        for _, box in enumerate(boxes):
            x, y, w, h = box.x, box.y, box.w, box.h
            if w * h <= self.nonMajorBoxSuppressionMaxRatio * maxTextBoxArea:
                break
            maxTextBoxRawHeight = max(maxTextBoxRawHeight, box.rawH)
            mask[y:y+h, x:x+w] = 255

        dialogVal: float = np.mean(mask)

        return mask, dialogVal, mask, maxTextBoxArea, maxTextBoxRawHeight
