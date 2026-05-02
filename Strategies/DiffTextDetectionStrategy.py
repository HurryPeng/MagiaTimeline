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
    class DialogFeature:
        image: cv.Mat
        mask: cv.Mat
        time: str
        maxTextBoxShortSide: int

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
    def genOcrEngine() -> paddleocr.TextDetection:
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

        self.ocr = DiffTextDetectionStrategy.genOcrEngine()

        self.rectangles: collections.OrderedDict[str, AbstractRectangle] = collections.OrderedDict()
        self.rectangles["dialogRect"] = RatioRectangle(contentRect, *config["dialogRect"])

        self.featureThreshold: float = config["featureThreshold"]
        self.boxVerticalExpansion: float = config["boxVerticalExpansion"]
        self.smallTextRawShortSideThreshold: float = 60.0
        self.smallTextBoxShortSideThreshold: float = self.smallTextRawShortSideThreshold * (1.0 + 2.0 * self.boxVerticalExpansion)
        self.nonMajorBoxSuppressionMaxRatio: float = config["nonMajorBoxSuppressionMaxRatio"]
        self.colourTolerance: int = config["colourTolerance"]
        self.minMaskIou: float = 0.5
        self.minOcrIou: float = 0.10
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
            self.log.write("time0,time1,merge,level,reason,maskIou,diffRate,cc,pcWarpDist,warpDist,sobelIou,postInpaintCommonEdgeRate,removedCommonEdgeRate,ocrIou,isTypewriterCandidate,isTypewriterRevoked,oldXCoverage,oldMaxTextBoxShortSide,newMaxTextBoxShortSide,isSmallTextComparison\n")
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
        oldMaxTextBoxShortSide = oldFeature.maxTextBoxShortSide
        newImage = newFeature.image
        newMask = newFeature.mask
        newTimeStr = newFeature.time
        newMaxTextBoxShortSide = newFeature.maxTextBoxShortSide
        isSmallText = (
            oldMaxTextBoxShortSide > 0 and
            newMaxTextBoxShortSide > 0 and
            max(oldMaxTextBoxShortSide, newMaxTextBoxShortSide) <= self.smallTextBoxShortSideThreshold
        )

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

        # Quick mask iou check before performing ocr on the intersection of the two images
        intersectMask = cv.bitwise_and(oldMask, newMask)
        unionMask = cv.bitwise_or(oldMask, newMask)
        intersectArea = np.sum(intersectMask) / 255
        unionArea = np.sum(unionMask) / 255
        
        if unionArea == 0:
            if self.debugLevel == 1:
                # time0,time1,merge,level,reason,sobelIou,postInpaintCommonEdgeRate,removedCommonEdgeRate,ocrIou
                self.log.write(f"{oldTimeStr},{newTimeStr},False,0,empty mask,0,0,0,0,0,0,0,0,0,False,False,0.0,{oldMaxTextBoxShortSide},{newMaxTextBoxShortSide},{isSmallText}\n")
                saveFrames()
            return False
        
        maskIou = intersectArea / unionArea

        isTypewriterCandidate = False
        isTypewriterRevoked = False
        oldXCoverage = 0.0

        if self.enableTypewriter:
            oldArea = np.sum(oldMask) / 255
            newArea = np.sum(newMask) / 255
            oldXWidth = self.xProjectionCoverage(oldMask)
            intersectXWidth = self.xProjectionCoverage(intersectMask)
            oldXCoverage = intersectXWidth / oldXWidth if oldXWidth > 0 else 0.0
            areaRatio = newArea / oldArea if oldArea > 0 else 0.0
            if oldXCoverage >= self.typewriterXCoverageThreshold and areaRatio > self.typewriterAreaRatioThreshold:
                isTypewriterCandidate = True

        if self.enableShortCircuit:
            if maskIou < self.minMaskIou and not isTypewriterCandidate:
                if self.debugLevel == 1:
                    self.log.write(f"{oldTimeStr},{newTimeStr},False,1,mask iou too low,{maskIou},0,0,0,0,0,0,0,0,{isTypewriterCandidate},{isTypewriterRevoked},{oldXCoverage},{oldMaxTextBoxShortSide},{newMaxTextBoxShortSide},{isSmallText}\n")
                    saveFrames()
                    saveExtra("2oldMask", oldMask)
                    saveExtra("3newMask", newMask)
                return False
        
        self.statDecideFeatureMergeDiff += 1
        
        diffMask: cv.Mat = rgbDiffMask(oldImage, newImage, self.colourTolerance)
        # The rate of pixels that are close enough
        diffArea = np.sum(diffMask) / 255
        diffRate = diffArea / unionArea
        if self.enableShortCircuit:
            if diffRate < 0.05:
                if self.debugLevel == 1:
                    self.log.write(f"{oldTimeStr},{newTimeStr},True,1,diff rate too low,{maskIou},{diffRate},0,0,0,0,0,0,0,{isTypewriterCandidate},{isTypewriterRevoked},{oldXCoverage},{oldMaxTextBoxShortSide},{newMaxTextBoxShortSide},{isSmallText}\n")
                    saveFrames()
                return True
        
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
        (shiftX, shiftY), response = phaseCorrelateMaxRes(
            src1=newImageSobelMaskedF32,
            src2=oldImageSobelMaskedF32,
            maxResolution=1800
        )
        if response > 0.1:
            warp = np.array([[1, 0, shiftX], [0, 1, shiftY]], dtype=np.float32)
        pcWarpDist = np.linalg.norm(warp[0:2, 2])
        if pcWarpDist > 1.5 and pcWarpDist < 50 and cc < 0.99:
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
        if warpDist > 1 and warpDist < 50 and cc > ccInit:
            warpedImage = cv.warpAffine(newImage, warp, (newImage.shape[1], newImage.shape[0]), flags=cv.INTER_LINEAR)
            warpedMask = cv.warpAffine(unionMask, warp, (newImage.shape[1], newImage.shape[0]), flags=cv.INTER_LINEAR)
            intersectMask = cv.bitwise_and(oldMask, warpedMask)
            unionMask = cv.bitwise_or(oldMask, warpedMask)
            if isTypewriterCandidate:
                oldXWidth = self.xProjectionCoverage(oldMask)
                intersectXWidth = self.xProjectionCoverage(intersectMask)
                oldXCoverage = intersectXWidth / oldXWidth if oldXWidth > 0 else 0.0
                if oldXCoverage < self.typewriterXCoverageThreshold:
                    isTypewriterCandidate = False
                    isTypewriterRevoked = True

        if isTypewriterCandidate:
            czNonZero = cv.findNonZero(intersectMask)
            if czNonZero is not None:
                _, _, _, _czHeight = cv.boundingRect(czNonZero)
                erosionRadius = max(1, int(_czHeight * self.boxVerticalExpansion / (1.0 + 2.0 * self.boxVerticalExpansion)))
                erosionRadius = int(erosionRadius * 1.5) # erode even more in order not to leak strokes in
                k = 2 * erosionRadius + 1
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
        iouIntersectSobelBinMasked = cv.bitwise_and(intersectSobelBin, iouUnionSobelBinMasked)
        commonEdgeBaseArea = np.sum(iouUnionSobelBinMasked) + 1e-6
        sobelIou = np.sum(iouIntersectSobelBinMasked) / commonEdgeBaseArea

        if self.enableShortCircuit:
            if sobelIou > 0.9:
                if self.debugLevel == 1:
                    self.log.write(f"{oldTimeStr},{newTimeStr},True,3,sobel iou too high,{maskIou},{diffRate},{cc},{pcWarpDist},{warpDist},{sobelIou},0,0,0,{isTypewriterCandidate},{isTypewriterRevoked},{oldXCoverage},{oldMaxTextBoxShortSide},{newMaxTextBoxShortSide},{isSmallText}\n")
                    saveFrames()
                    saveExtra("2oldSobel", oldImageSobelBin)
                    saveExtra("3newSobel", warpedImageSobelBin)
                    saveExtra("4unionSobel", unionSobelBinMasked)
                    saveExtra("5intersectSobel", intersectSobelBinMasked)
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
        intersectPostInpaintSobelMasked = cv.bitwise_and(intersectPostInpaintSobel, iouUnionSobelBinMasked)
        postInpaintCommonEdgeRate = np.sum(intersectPostInpaintSobelMasked) / commonEdgeBaseArea

        removedCommonEdgeRate = sobelIou - postInpaintCommonEdgeRate

        if self.enableShortCircuit:
            if removedCommonEdgeRate > 0.7:
                if self.debugLevel == 1:
                    self.log.write(f"{oldTimeStr},{newTimeStr},True,3,common edge removal too high,{maskIou},{diffRate},{cc},{pcWarpDist},{warpDist},{sobelIou},{postInpaintCommonEdgeRate},{removedCommonEdgeRate},0,{isTypewriterCandidate},{isTypewriterRevoked},{oldXCoverage},{oldMaxTextBoxShortSide},{newMaxTextBoxShortSide},{isSmallText}\n")
                    saveFrames()
                    saveExtra("2oldSobel", oldImageSobelBin)
                    saveExtra("3newSobel", warpedImageSobelBin)
                    saveExtra("4unionSobel", unionSobelBinMasked)
                    saveExtra("5intersectSobel", intersectSobelBinMasked)
                    saveExtra("6diffMask", diffMask)
                    saveExtra("7inpaintMask", inpaintMask)
                    saveExtra("8inpaint", warpedImageInpaint)
                return True
            if removedCommonEdgeRate < 0.2:
                if self.debugLevel == 1:
                    self.log.write(f"{oldTimeStr},{newTimeStr},False,3,common edge removal too low,{maskIou},{diffRate},{cc},{pcWarpDist},{warpDist},{sobelIou},{postInpaintCommonEdgeRate},{removedCommonEdgeRate},0,{isTypewriterCandidate},{isTypewriterRevoked},{oldXCoverage},{oldMaxTextBoxShortSide},{newMaxTextBoxShortSide},{isSmallText}\n")
                    saveFrames()
                    saveExtra("2oldSobel", oldImageSobelBin)
                    saveExtra("3newSobel", warpedImageSobelBin)
                    saveExtra("4unionSobel", unionSobelBinMasked)
                    saveExtra("5intersectSobel", intersectSobelBinMasked)
                    saveExtra("6diffMask", diffMask)
                    saveExtra("7inpaintMask", inpaintMask)
                    saveExtra("8inpaint", warpedImageInpaint)
                return False

        # Stage 7: Post-inpaint text detection
        self.statDecideFeatureMergeOCR += 1

        ocrMask, _, _, _ = self.ocrPass(warpedImageInpaint)
        ocrIntersectMask = cv.bitwise_and(ocrMask, iouMask)
        ocrIntersectVal = np.mean(ocrIntersectMask)
        ocrIntersectArea = np.sum(ocrIntersectMask) / 255
        ocrIou = ocrIntersectArea / iouArea if iouArea > 0 else 0.0
        ocrDecision: bool = ocrIntersectVal < self.featureThreshold or ocrIou < self.minOcrIou
        if self.debugLevel == 1:
            self.log.write(f"{oldTimeStr},{newTimeStr},{ocrDecision},4,ocr decision,{maskIou},{diffRate},{cc},{pcWarpDist},{warpDist},{sobelIou},{postInpaintCommonEdgeRate},{removedCommonEdgeRate},{ocrIou},{isTypewriterCandidate},{isTypewriterRevoked},{oldXCoverage},{oldMaxTextBoxShortSide},{newMaxTextBoxShortSide},{isSmallText}\n")
            saveFrames()
            saveExtra("2oldSobel", oldImageSobelBin)
            saveExtra("3newSobel", warpedImageSobelBin)
            saveExtra("4unionSobel", unionSobelBinMasked)
            saveExtra("5intersectSobel", intersectSobelBinMasked)
            saveExtra("6diffMask", diffMask)
            saveExtra("7inpaintMask", inpaintMask)
            saveExtra("8inpaint", warpedImageInpaint)
            saveExtra("9ocrMask", ocrMask)
        return ocrDecision
    
    def aggregateFeatures(self, features: typing.List[typing.Any]) -> typing.Any:
        # simply return the last feature
        return features[-1]

    def isFpNonEmpty(self, fp: FramePoint) -> bool:
        return fp.getFlag(DiffTextDetectionStrategy.FlagIndex.Dialog)

    def getFpFeature(self, fp: FramePoint) -> typing.Any:
        return fp.getFlag(DiffTextDetectionStrategy.FlagIndex.DialogFeat)

    def freeFpFeature(self, fp: FramePoint) -> None:
        fp.setFlag(DiffTextDetectionStrategy.FlagIndex.DialogFeat, None)

    def cutExtraJobFrame(self, frame: cv.Mat) -> cv.Mat:
        return self.dialogRect.cutRoi(frame)
    
    def detectTextBoxes(self, frame: cv.Mat) -> typing.List[typing.Tuple[int, int, int, int]]:
        imgH, imgW = frame.shape[:2]

        scaleDown = 1
        while imgH // scaleDown > 960 or imgW // scaleDown > 960:
            scaleDown *= 2

        if scaleDown > 1:
            frame = cv.resize(frame, (imgW // scaleDown, imgH // scaleDown), interpolation=cv.INTER_LINEAR)

        result = self.ocr.predict(frame)
        result = result[0]
        dtPolys: typing.List[np.ndarray] = result["dt_polys"]
        dtScores: typing.List[float] = result["dt_scores"]
        n = len(dtPolys)
        assert n == len(dtScores)
        
        if n == 0:
            return []

        boxes = []
        for i in range(n):
            wordInfo = dtPolys[i]
            confidence = dtScores[i]
            wordInfo = np.array(wordInfo, np.int32)
            wordInfo *= scaleDown
            x0, y0 = wordInfo[0]
            x1, y1 = wordInfo[1]
            x2, y2 = wordInfo[2]
            x3, y3 = wordInfo[3]
            angle0 = np.arctan2(y1 - y0, x1 - x0)
            angle3 = np.arctan2(y2 - y3, x2 - x3)
            angle = (angle0 + angle3) / 2
            if np.abs(angle) > np.pi / 180 * 3:
                continue

            x0, y0, w0, h0 = cv.boundingRect(wordInfo)
            expand = int(h0 * self.boxVerticalExpansion)
            x = max(0, x0 - expand)
            y = max(0, y0 - expand)
            w = min(imgW, w0 + 2 * expand)
            h = min(imgH, h0 + 2 * expand)
            wordInfo = (x, y, w, h)
            boxes.append(wordInfo)

        return boxes

    def cvPassDialog(self, frame: cv.Mat, framePoint: FramePoint) -> bool:

        image = self.cutExtraJobFrame(frame)

        mask, dialogVal, debugFrame, maxTextBoxShortSide = self.ocrPass(image)

        hasDialog = dialogVal > self.featureThreshold

        feat: DiffTextDetectionStrategy.DialogFeature | None = None
        if hasDialog:
            timeString = framePoint.timeString()
            feat = DiffTextDetectionStrategy.DialogFeature(
                image=image,
                mask=mask,
                time=timeString,
                maxTextBoxShortSide=maxTextBoxShortSide,
            )

        framePoint.setFlag(DiffTextDetectionStrategy.FlagIndex.Dialog, hasDialog)
        framePoint.setFlag(DiffTextDetectionStrategy.FlagIndex.DialogVal, dialogVal)
        framePoint.setFlag(DiffTextDetectionStrategy.FlagIndex.DialogFeat, feat)

        return False

    def ocrPass(self, frame: cv.Mat) -> typing.Tuple[cv.Mat, float, cv.Mat | None, int]:
        # returns mask, dialogVal, debugFrame, maxTextBoxShortSide

        boxes = self.detectTextBoxes(frame)

        mask: cv.Mat = np.zeros_like(frame[:, :, 0])
        boxes = sorted(boxes, key=lambda box: box[2] * box[3], reverse=True)
        maxTextBoxShortSide = max((min(w, h) for _, _, w, h in boxes), default=0)
        biggestBoxArea = boxes[0][2] * boxes[0][3] if boxes else 0
        for rank, box in enumerate(boxes):
            x, y, w, h = box
            if w * h <= self.nonMajorBoxSuppressionMaxRatio * biggestBoxArea:
                break
            mask[y:y+h, x:x+w] = 255

        dialogVal: float = np.mean(mask)

        return mask, dialogVal, mask, maxTextBoxShortSide
