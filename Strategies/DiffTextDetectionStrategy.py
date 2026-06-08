from __future__ import annotations

import atexit
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

    @dataclasses.dataclass
    class MergeContext:
        """Accumulates intermediate results across merge decision stages."""
        # Input features
        oldImage: cv.Mat
        newImage: cv.Mat
        oldMask: cv.Mat
        newMask: cv.Mat
        oldTimeStr: str
        newTimeStr: str
        oldMaxTextBoxArea: int
        newMaxTextBoxArea: int
        oldMaxTextBoxRawHeight: int
        newMaxTextBoxRawHeight: int
        isSmallText: bool

        # Mask IoU
        intersectMask: cv.Mat = None
        unionMask: cv.Mat = None
        intersectArea: int = 0
        unionArea: int = 0
        maskIou: float = 0.0

        # Typewriter
        isTypewriterCandidate: bool = False
        isTypewriterRevoked: bool = False
        oldXCoverage: float = 0.0
        oldXWidth: int = 0

        # Diff rate
        diffMask: cv.Mat = None
        diffRate: float = 0.0

        # ECC alignment
        warp: np.ndarray = None
        warpedImage: cv.Mat = None
        eccCorrelation: float = 0.0
        phaseCorrelationDist: float = 0.0
        warpDist: float = 0.0
        maxWarpDist: float = 0.0
        oldImageSobel: cv.Mat = None

        # iouMask
        iouMask: cv.Mat = None
        iouArea: int = 0

        # Sobel features
        warpedImageSobel: cv.Mat = None
        oldImageSobelBin: cv.Mat = None
        warpedImageSobelBin: cv.Mat = None
        oldImageSobelBinDilate: cv.Mat = None
        unionSobelBinMasked: cv.Mat = None
        intersectSobelBinMasked: cv.Mat = None
        intersectSobelBinMaskedDilate: cv.Mat = None
        unionSobelBin: cv.Mat = None
        iouUnionSobelBinMasked: cv.Mat = None
        commonEdgeBaseMask: cv.Mat = None
        commonEdgeBaseArea: float = 0.0
        commonEdgeBaseAreaSafe: float = 0.0
        unionEdgeBaseArea: float = 0.0
        sobelIou: float = 0.0
        sobelEdgeDensity: float = 0.0
        oneSidedEdgeRate: float = 0.0

        # Inpainting
        warpedImageInpaint: cv.Mat = None
        inpaintMask: cv.Mat = None

        # Common edge removal
        postInpaintCommonEdgeRate: float = 0.0
        removedCommonEdgeRate: float = 0.0
        postInpaintUnionEdgeRate: float = 0.0
        removedUnionEdgeRate: float = 0.0
        commonUnionRemovalGap: float = 0.0

        # OCR / decision
        ocrSaysMerge: bool = False
        finalDecision: bool = False
        decisionReason: str = ""
        ocrIou: float = 0.0
        postInpaintProbValue: float = 0.0
        inpaintOcrMask: cv.Mat = None
        inpaintProbInt8: cv.Mat = None

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

        # Small-text / mask IOU thresholds (decideFeatureMerge early checks)
        self.smallTextBoxAreaThreshold: int = 60000
        self.minMaskIou: float = 0.5

        # Typewriter detection (decideFeatureMerge typewriter block)
        self.enableTypewriter: bool = config["enableTypewriter"]
        self.typewriterXCoverageThreshold: float = config["typewriterXCoverageThreshold"]
        self.typewriterAreaRatioThreshold: float = config["typewriterAreaRatioThreshold"]
        self.typewriterErosionFactor: float = 1.5

        # Diff-rate shortcut (decideFeatureMerge diff-rate block)
        self.colourTolerance: int = config["colourTolerance"]
        self.enableShortCircuit: bool = config["enableShortCircuit"]
        self.diffRateMergeThreshold: float = 0.05

        # ECC / warp alignment (decideFeatureMerge ECC block)
        self.boxExpansion: float = config["boxExpansion"]
        self.maxWarpTextBoxMinSideRatio: float = 0.60
        self.maxWarpResolution: int = 1800
        self.phaseCorrelationResponseMin: float = 0.1
        self.phaseCorrelationDistanceMin: float = 1.5
        self.eccCorrelationLow: float = 0.99
        self.eccCriteria = (cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 10, 0.01)
        self.warpDistanceMin: int = 1

        # Sobel IOU filtering (decideFeatureMerge sobel block)
        self.sobelBinarizeThreshold: int = 32
        self.sobelIouMergeThreshold: float = 0.9
        self.sobelShortcutMaxEdgeDensity: float = 0.45

        # Inpainting (decideFeatureMerge inpainting block)
        self.inpaintEdgeBlur = (21, 21)
        self.inpaintSmoothBlur = (11, 11)

        # Common-edge merge/split (decideFeatureMerge post-inpaint edge block)
        self.commonEdgeRemovalSplitThreshold: float = 0.25
        self.commonEdgeRemovalMergeThreshold: float = 0.70
        self.commonEdgeRemovalMergeMinSobelIou: float = 0.70
        self.commonEdgeRemovalMergeMaxOneSidedEdgeRate: float = 0.30
        self.commonEdgeRemovalMergeMinUnionEdgeRemovalRate: float = 0.80

        # OCR / heatmap decision (decideFeatureMerge final decision)
        self.heatmapOverrideThreshold: float = 0.05
        self.featureThreshold: float = config["featureThreshold"]
        self.minOcrIou: float = 0.10
        self.probMapThreshold: float = 0.5

        # Text-detection post-processing (detectTextBoxesAndProbMap)
        self.nonMajorBoxSuppressionMaxRatio: float = config["nonMajorBoxSuppressionMaxRatio"]
        self.maxTextAngleRad: float = np.pi / 180 * 3
        self.maxDetectionSide: int = 960

        # IIR / debug (specIirPasses / decideFeatureMerge debug)
        self.iirPassDenoiseMinTime: int = config["iirPassDenoiseMinTime"]
        self.debugLevel: int = config["debugLevel"]

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
            logFile = self.log
            def closeLogFile():
                if not logFile.closed:
                    logFile.close()
            atexit.register(closeLogFile)

    @classmethod
    def getFlagIndexType(cls) -> typing.Type[AbstractFlagIndex]:
        return cls.FlagIndex
    
    @staticmethod
    def isEmptyFeature(feature: DialogFeature | None) -> bool:
        return feature is None

    def ellipseKernel(self, k: int) -> cv.Mat:
        return cv.getStructuringElement(cv.MORPH_ELLIPSE, (k, k))

    @staticmethod
    def xProjectionSpan(mask: cv.Mat) -> int:
        return int(np.count_nonzero(np.any(mask > 0, axis=0)))

    @staticmethod
    def yProjectionSpan(mask: cv.Mat) -> int:
        return int(np.count_nonzero(np.any(mask > 0, axis=1)))


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
    
    def computeMaskIoU(self, ctx: DiffTextDetectionStrategy.MergeContext) -> None:
        ctx.intersectMask = cv.bitwise_and(ctx.oldMask, ctx.newMask)
        ctx.unionMask = cv.bitwise_or(ctx.oldMask, ctx.newMask)
        ctx.intersectArea = cv.countNonZero(ctx.intersectMask)
        ctx.unionArea = cv.countNonZero(ctx.unionMask)
        ctx.maskIou = ctx.intersectArea / ctx.unionArea if ctx.unionArea > 0 else 0.0

    def checkTypewriter(self, ctx: DiffTextDetectionStrategy.MergeContext) -> None:
        oldArea = cv.countNonZero(ctx.oldMask)
        newArea = cv.countNonZero(ctx.newMask)
        ctx.oldXWidth = self.xProjectionSpan(ctx.oldMask)
        intersectXWidth = self.xProjectionSpan(ctx.intersectMask)
        ctx.oldXCoverage = intersectXWidth / ctx.oldXWidth if ctx.oldXWidth > 0 else 0.0
        areaRatio = newArea / oldArea if oldArea > 0 else 0.0
        ctx.isTypewriterCandidate = (
            ctx.oldXCoverage >= self.typewriterXCoverageThreshold
            and areaRatio > self.typewriterAreaRatioThreshold
        )

    def computeDiffRate(self, ctx: DiffTextDetectionStrategy.MergeContext) -> None:
        ctx.diffMask = rgbDiffMask(ctx.oldImage, ctx.newImage, self.colourTolerance)
        diffArea = cv.countNonZero(ctx.diffMask)
        ctx.diffRate = diffArea / ctx.unionArea

    def computeEccAlignment(self, ctx: DiffTextDetectionStrategy.MergeContext) -> None:
        ctx.oldImageSobel = rgbSobel(ctx.oldImage, 1)
        newImageSobel = rgbSobel(ctx.newImage, 1)

        self.statDecideFeatureMergeComputeECC += 1

        ctx.warp = np.eye(2, 3, dtype=np.float32)
        eccCorrelationInit = cv.computeECC(
            templateImage=newImageSobel, inputImage=ctx.oldImageSobel, inputMask=ctx.unionMask
        )
        ctx.eccCorrelation = eccCorrelationInit

        oldImageSobelMasked = cv.bitwise_and(ctx.oldImageSobel, ctx.oldImageSobel, mask=ctx.oldMask)
        newImageSobelMasked = cv.bitwise_and(newImageSobel, newImageSobel, mask=ctx.newMask)
        oldImageSobelMaskedF32 = np.float32(oldImageSobelMasked)
        newImageSobelMaskedF32 = np.float32(newImageSobelMasked)
        oldTextMinSide = min(self.xProjectionSpan(ctx.oldMask), self.yProjectionSpan(ctx.oldMask))
        newTextMinSide = min(self.xProjectionSpan(ctx.newMask), self.yProjectionSpan(ctx.newMask))
        rawTextMinSide = min(oldTextMinSide, newTextMinSide) / (1.0 + 2.0 * self.boxExpansion)
        maxWarpDist = min(50, self.maxWarpTextBoxMinSideRatio * rawTextMinSide)

        (shiftX, shiftY), response = phaseCorrelateMaxRes(
            src1=newImageSobelMaskedF32, src2=oldImageSobelMaskedF32,
            maxResolution=self.maxWarpResolution
        )
        if response > self.phaseCorrelationResponseMin:
            ctx.warp = np.array([[1, 0, shiftX], [0, 1, shiftY]], dtype=np.float32)
        ctx.phaseCorrelationDist = np.linalg.norm(ctx.warp[0:2, 2])

        if ctx.phaseCorrelationDist > self.phaseCorrelationDistanceMin and ctx.phaseCorrelationDist < maxWarpDist and ctx.eccCorrelation < self.eccCorrelationLow:
            try:
                self.statDecideFeatureMergeFindTransformECC += 1
                ctx.eccCorrelation, ctx.warp = cv.findTransformECC(
                    templateImage=newImageSobel, inputImage=ctx.oldImageSobel,
                    warpMatrix=ctx.warp, motionType=cv.MOTION_TRANSLATION,
                    criteria=self.eccCriteria, inputMask=ctx.unionMask,
                )
            except cv.error:
                ctx.eccCorrelation = 0
                ctx.warp = np.eye(2, 3, dtype=np.float32)

        ctx.warpedImage = ctx.newImage
        ctx.warpDist = np.linalg.norm(ctx.warp[0:2, 2])
        ctx.maxWarpDist = maxWarpDist
        ctx.isTypewriterRevoked = False
        if ctx.warpDist > self.warpDistanceMin and ctx.warpDist < maxWarpDist and ctx.eccCorrelation > eccCorrelationInit:
            ctx.warpedImage = cv.warpAffine(ctx.newImage, ctx.warp, (ctx.newImage.shape[1], ctx.newImage.shape[0]), flags=cv.INTER_LINEAR)
            warpedMask = cv.warpAffine(ctx.unionMask, ctx.warp, (ctx.newImage.shape[1], ctx.newImage.shape[0]), flags=cv.INTER_LINEAR)
            ctx.intersectMask = cv.bitwise_and(ctx.oldMask, warpedMask)
            ctx.unionMask = cv.bitwise_or(ctx.oldMask, warpedMask)
            if ctx.isTypewriterCandidate:
                intersectXWidth = self.xProjectionSpan(ctx.intersectMask)
                ctx.oldXCoverage = intersectXWidth / ctx.oldXWidth if ctx.oldXWidth > 0 else 0.0
                if ctx.oldXCoverage < self.typewriterXCoverageThreshold:
                    ctx.isTypewriterCandidate = False
                    ctx.isTypewriterRevoked = True
        else:
            ctx.intersectMask = cv.bitwise_and(ctx.oldMask, ctx.newMask)
            ctx.unionMask = cv.bitwise_or(ctx.oldMask, ctx.newMask)

    def computeIouMask(self, ctx: DiffTextDetectionStrategy.MergeContext) -> None:
        if ctx.isTypewriterCandidate:
            intersectPoints = cv.findNonZero(ctx.intersectMask)
            if intersectPoints is not None:
                typewriterTextHeights = [
                    height for height in [ctx.oldMaxTextBoxRawHeight, ctx.newMaxTextBoxRawHeight]
                    if height > 0
                ]
                typewriterTextHeight = min(typewriterTextHeights) if typewriterTextHeights else cv.boundingRect(intersectPoints)[3]
                erosionRadius = max(1, int(typewriterTextHeight * self.boxExpansion))
                erosionRadius = int(erosionRadius * self.typewriterErosionFactor) # erode even more in order not to leak strokes in
                k = max(3, 2 * erosionRadius + 1)
                ctx.iouMask = cv.morphologyEx(ctx.intersectMask, cv.MORPH_ERODE, self.ellipseKernel(k))
            else:
                ctx.iouMask = ctx.intersectMask
            ctx.iouArea = cv.countNonZero(ctx.iouMask)
        else:
            ctx.iouMask = ctx.unionMask
            ctx.iouArea = cv.countNonZero(ctx.iouMask)

    def computeSobelFeatures(self, ctx: DiffTextDetectionStrategy.MergeContext) -> None:
        ctx.warpedImageSobel = rgbSobel(ctx.warpedImage, 1)
        ctx.oldImageSobelBin = cv.threshold(ctx.oldImageSobel, self.sobelBinarizeThreshold, 255, cv.THRESH_BINARY)[1]
        ctx.warpedImageSobelBin = cv.threshold(ctx.warpedImageSobel, self.sobelBinarizeThreshold, 255, cv.THRESH_BINARY)[1]

        ctx.oldImageSobelBinDilate = cv.morphologyEx(ctx.oldImageSobelBin, cv.MORPH_DILATE, self.ellipseKernel(3))
        warpedImageSobelBinDilate = cv.morphologyEx(ctx.warpedImageSobelBin, cv.MORPH_DILATE, self.ellipseKernel(3))
        ctx.unionSobelBin = cv.bitwise_or(ctx.oldImageSobelBin, ctx.warpedImageSobelBin)
        ctx.unionSobelBinMasked = cv.bitwise_and(ctx.unionSobelBin, ctx.unionMask)
        intersectSobelBin = cv.bitwise_and(ctx.oldImageSobelBinDilate, warpedImageSobelBinDilate)
        ctx.intersectSobelBinMasked = cv.bitwise_and(intersectSobelBin, ctx.unionSobelBinMasked)
        ctx.intersectSobelBinMaskedDilate = cv.morphologyEx(ctx.intersectSobelBinMasked, cv.MORPH_DILATE, self.ellipseKernel(3))
        ctx.iouUnionSobelBinMasked = cv.bitwise_and(ctx.unionSobelBin, ctx.iouMask)
        oldOnlySobelBin = cv.bitwise_and(ctx.oldImageSobelBin, cv.bitwise_not(warpedImageSobelBinDilate))
        warpedOnlySobelBin = cv.bitwise_and(ctx.warpedImageSobelBin, cv.bitwise_not(ctx.oldImageSobelBinDilate))
        oneSidedSobelBin = cv.bitwise_or(oldOnlySobelBin, warpedOnlySobelBin)
        oneSidedSobelBinMasked = cv.bitwise_and(oneSidedSobelBin, ctx.iouMask)
        ctx.commonEdgeBaseMask = cv.bitwise_and(intersectSobelBin, ctx.iouUnionSobelBinMasked)
        ctx.unionEdgeBaseArea = np.sum(ctx.iouUnionSobelBinMasked) + 1e-6
        ctx.commonEdgeBaseArea = np.sum(ctx.commonEdgeBaseMask)
        ctx.commonEdgeBaseAreaSafe = ctx.commonEdgeBaseArea + 1e-6
        ctx.sobelIou = ctx.commonEdgeBaseArea / ctx.unionEdgeBaseArea
        ctx.sobelEdgeDensity = cv.countNonZero(ctx.iouUnionSobelBinMasked) / (ctx.iouArea + 1e-6)
        ctx.oneSidedEdgeRate = cv.countNonZero(oneSidedSobelBinMasked) / (cv.countNonZero(ctx.iouUnionSobelBinMasked) + 1e-6)

    def runInpainting(self, ctx: DiffTextDetectionStrategy.MergeContext) -> None:
        self.statDecideFeatureMergeInpaint += 1

        diffMask = rgbDiffMask(ctx.oldImage, ctx.warpedImage, self.colourTolerance)
        inpaintMask = cv.bitwise_not(diffMask)

        inpaintMaskEdge = cv.morphologyEx(inpaintMask, cv.MORPH_GRADIENT, self.ellipseKernel(3))
        cv.copyTo(src=ctx.intersectSobelBinMaskedDilate, dst=inpaintMask, mask=inpaintMaskEdge)

        inpaintMaskAndIntersectSobel = cv.bitwise_and(inpaintMask, ctx.intersectSobelBinMasked)
        cv.copyTo(src=inpaintMaskAndIntersectSobel, dst=inpaintMask, mask=ctx.unionSobelBin)

        if not ctx.isSmallText:
            inpaintMaskDilate = cv.morphologyEx(inpaintMask, cv.MORPH_DILATE, self.ellipseKernel(3))
            inpaintMaskDilateErode = cv.morphologyEx(inpaintMaskDilate, cv.MORPH_ERODE, self.ellipseKernel(5))
            inpaintMask = cv.bitwise_or(inpaintMask, inpaintMaskDilateErode)

        inpaintMask = cv.bitwise_and(inpaintMask, ctx.unionMask)

        inpaintBaseImage = ctx.warpedImage
        if ctx.isSmallText:
            useOldPixel = (ctx.oldImageSobel >= ctx.warpedImageSobel)[:, :, np.newaxis]
            inpaintBaseImage = np.where(useOldPixel, ctx.oldImage, ctx.warpedImage).astype(np.uint8)

        warpedImageNoInpaintMask = cv.bitwise_and(inpaintBaseImage, inpaintBaseImage, mask=cv.bitwise_not(inpaintMask))
        warpedImageNoInpaintMaskBlur = cv.stackBlur(warpedImageNoInpaintMask, self.inpaintEdgeBlur)
        warpedImageNoInpaintMaskDenom = cv.stackBlur(cv.bitwise_not(inpaintMask), self.inpaintEdgeBlur)
        warpedImageNoInpaintMaskDenom[warpedImageNoInpaintMaskDenom == 0] = 1
        warpedImageNoInpaintMaskDenom = cv.merge([warpedImageNoInpaintMaskDenom] * 3)
        inpaintBase = cv.divide(warpedImageNoInpaintMaskBlur, warpedImageNoInpaintMaskDenom, scale=256, dtype=cv.CV_8U)

        inpaintIntermediate = cv.inpaint(inpaintBase, inpaintMask, 1, cv.INPAINT_TELEA)

        ctx.warpedImageInpaint = inpaintBaseImage.copy()
        cv.copyTo(src=inpaintIntermediate, dst=ctx.warpedImageInpaint, mask=inpaintMask)

        warpedImageInpaintBlur = cv.stackBlur(ctx.warpedImageInpaint, self.inpaintSmoothBlur)
        cv.copyTo(src=warpedImageInpaintBlur, dst=ctx.warpedImageInpaint, mask=inpaintMask)

        ctx.inpaintMask = inpaintMask

    def computeCommonEdgeRemoval(self, ctx: DiffTextDetectionStrategy.MergeContext) -> None:
        warpedImageInpaintSobel = rgbSobel(ctx.warpedImageInpaint, 1)
        warpedImageInpaintSobelBin = cv.threshold(warpedImageInpaintSobel, self.sobelBinarizeThreshold, 255, cv.THRESH_BINARY)[1]
        warpedImageInpaintSobelBinDilate = cv.morphologyEx(warpedImageInpaintSobelBin, cv.MORPH_DILATE, self.ellipseKernel(5))
        intersectPostInpaintSobel = cv.bitwise_and(ctx.oldImageSobelBinDilate, warpedImageInpaintSobelBinDilate)
        remainingCommonEdgeMask = cv.bitwise_and(intersectPostInpaintSobel, ctx.commonEdgeBaseMask)
        remainingUnionEdgeMask = cv.bitwise_and(warpedImageInpaintSobelBinDilate, ctx.iouUnionSobelBinMasked)

        if ctx.commonEdgeBaseArea > 0:
            ctx.postInpaintCommonEdgeRate = np.sum(remainingCommonEdgeMask) / ctx.commonEdgeBaseAreaSafe
            ctx.removedCommonEdgeRate = 1.0 - ctx.postInpaintCommonEdgeRate
        else:
            ctx.postInpaintCommonEdgeRate = 0.0
            ctx.removedCommonEdgeRate = 0.0

        if ctx.unionEdgeBaseArea > 0:
            ctx.postInpaintUnionEdgeRate = np.sum(remainingUnionEdgeMask) / ctx.unionEdgeBaseArea
            ctx.removedUnionEdgeRate = 1.0 - ctx.postInpaintUnionEdgeRate
        else:
            ctx.postInpaintUnionEdgeRate = 0.0
            ctx.removedUnionEdgeRate = 0.0

        ctx.commonUnionRemovalGap = ctx.removedCommonEdgeRate - ctx.removedUnionEdgeRate

    def runOcrAndDecision(self, ctx: DiffTextDetectionStrategy.MergeContext) -> None:
        inpaintBoxes, inpaintProb = self.detectTextBoxesAndProbMap(ctx.warpedImageInpaint)
        ctx.inpaintProbInt8 = np.clip(inpaintProb * 255, 0, 255).astype(np.uint8)

        self.statDecideFeatureMergeOCR += 1
        ctx.inpaintOcrMask = np.zeros_like(ctx.iouMask)
        for box in inpaintBoxes:
            ctx.inpaintOcrMask[box.y:box.y+box.h, box.x:box.x+box.w] = 255

        ocrIntersectMask = cv.bitwise_and(ctx.inpaintOcrMask, ctx.iouMask)
        ocrIntersectVal = np.mean(ocrIntersectMask)
        ocrIntersectArea = cv.countNonZero(ocrIntersectMask)
        ctx.ocrIou = ocrIntersectArea / ctx.iouArea if ctx.iouArea > 0 else 0.0
        ctx.ocrSaysMerge = ocrIntersectVal < self.featureThreshold or ctx.ocrIou < self.minOcrIou

        ctx.postInpaintProbValue = 0.0
        iouMaskF = ctx.iouMask.astype(np.float32) / 255.0
        iouAreaF = np.sum(iouMaskF)
        if iouAreaF > 0:
            ctx.postInpaintProbValue = float(
                np.sum((inpaintProb > self.probMapThreshold).astype(np.float32) * iouMaskF) / iouAreaF
            )

        heatmapSplitOverride = False
        if ctx.postInpaintProbValue > self.heatmapOverrideThreshold:
            if ctx.ocrSaysMerge:
                heatmapSplitOverride = True

        if heatmapSplitOverride:
            ctx.finalDecision = False
            ctx.decisionReason = f"heatmap split override (highRate={ctx.postInpaintProbValue:.4f})"
        else:
            ctx.finalDecision = ctx.ocrSaysMerge
            ctx.decisionReason = "ocr decision"

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

        ctx = DiffTextDetectionStrategy.MergeContext(
            oldImage=oldImage, newImage=newImage,
            oldMask=oldMask, newMask=newMask,
            oldTimeStr=oldTimeStr, newTimeStr=newTimeStr,
            oldMaxTextBoxArea=oldMaxTextBoxArea, newMaxTextBoxArea=newMaxTextBoxArea,
            oldMaxTextBoxRawHeight=oldMaxTextBoxRawHeight, newMaxTextBoxRawHeight=newMaxTextBoxRawHeight,
            isSmallText=isSmallText,
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
                ctx.maskIou,
                ctx.diffRate,
                ctx.eccCorrelation,
                ctx.phaseCorrelationDist,
                ctx.warpDist,
                ctx.maxWarpDist,
                ctx.sobelIou,
                ctx.postInpaintCommonEdgeRate,
                ctx.removedCommonEdgeRate,
                ocrIou,
                ctx.isTypewriterCandidate,
                ctx.isTypewriterRevoked,
                ctx.oldXCoverage,
                oldMaxTextBoxArea,
                newMaxTextBoxArea,
                oldMaxTextBoxRawHeight,
                newMaxTextBoxRawHeight,
                isSmallText,
                ctx.oneSidedEdgeRate,
                ctx.sobelEdgeDensity,
                ctx.postInpaintUnionEdgeRate,
                ctx.removedUnionEdgeRate,
                ctx.commonEdgeBaseArea,
                ctx.unionEdgeBaseArea,
                ctx.commonUnionRemovalGap,
                ctx.postInpaintProbValue,
            ]
            self.log.write(",".join(str(value) for value in values) + "\n")
            saveFrames()
            for step, image in images:
                saveExtra(step, image)

        # Quick mask iou check before performing ocr on the intersection of the two images
        self.computeMaskIoU(ctx)
        if ctx.unionArea == 0:
            writeDebugDecision(False, 0, "empty mask")
            return False

        if self.enableTypewriter:
            self.checkTypewriter(ctx)

        self.statDecideFeatureMergeDiff += 1
        self.computeDiffRate(ctx)
        if self.enableShortCircuit:
            if ctx.diffRate < self.diffRateMergeThreshold:
                writeDebugDecision(True, 1, "diff rate too low")
                return True

            if ctx.maskIou < self.minMaskIou and not ctx.isTypewriterCandidate:
                writeDebugDecision(False, 1, "mask iou too low", images=[
                    ("2oldMask", oldMask),
                    ("3newMask", newMask),
                ])
                return False

        # ECC alignment
        self.computeEccAlignment(ctx)

        # Compute iouMask
        self.computeIouMask(ctx)

        # Sobel features
        self.computeSobelFeatures(ctx)

        if self.enableShortCircuit:
            if ctx.sobelIou > self.sobelIouMergeThreshold and ctx.sobelEdgeDensity < self.sobelShortcutMaxEdgeDensity:
                writeDebugDecision(True, 3, "sobel iou too high", images=[
                    ("2oldSobel", ctx.oldImageSobelBin),
                    ("3newSobel", ctx.warpedImageSobelBin),
                    ("4unionSobel", ctx.unionSobelBinMasked),
                    ("5intersectSobel", ctx.intersectSobelBinMasked),
                    ("6iouMask", ctx.iouMask),
                ])
                return True

        # Inpainting
        self.runInpainting(ctx)

        # Post-inpaint common edge removal filtering
        self.computeCommonEdgeRemoval(ctx)

        if self.enableShortCircuit:
            if ctx.commonEdgeBaseArea > 0 and ctx.removedCommonEdgeRate < self.commonEdgeRemovalSplitThreshold:
                writeDebugDecision(False, 3, "common edge removal too low", images=[
                    ("2oldSobel", ctx.oldImageSobelBin),
                    ("3newSobel", ctx.warpedImageSobelBin),
                    ("4unionSobel", ctx.unionSobelBinMasked),
                    ("5intersectSobel", ctx.intersectSobelBinMasked),
                    ("6diffMask", ctx.diffMask),
                    ("7inpaintMask", ctx.inpaintMask),
                    ("8inpaint", ctx.warpedImageInpaint),
                    ("10iouMask", ctx.iouMask),
                ])
                return False
            if (
                ctx.removedCommonEdgeRate > self.commonEdgeRemovalMergeThreshold and
                ctx.sobelIou > self.commonEdgeRemovalMergeMinSobelIou and
                ctx.oneSidedEdgeRate < self.commonEdgeRemovalMergeMaxOneSidedEdgeRate and
                ctx.removedUnionEdgeRate > self.commonEdgeRemovalMergeMinUnionEdgeRemovalRate
            ):
                writeDebugDecision(True, 3, "common edge removal too high", images=[
                    ("2oldSobel", ctx.oldImageSobelBin),
                    ("3newSobel", ctx.warpedImageSobelBin),
                    ("4unionSobel", ctx.unionSobelBinMasked),
                    ("5intersectSobel", ctx.intersectSobelBinMasked),
                    ("6diffMask", ctx.diffMask),
                    ("7inpaintMask", ctx.inpaintMask),
                    ("8inpaint", ctx.warpedImageInpaint),
                    ("10iouMask", ctx.iouMask),
                ])
                return True

        # OCR and decision
        self.runOcrAndDecision(ctx)

        writeDebugDecision(ctx.finalDecision, 4, ctx.decisionReason, ocrIou=ctx.ocrIou, images=[
            ("2oldSobel", ctx.oldImageSobelBin),
            ("3newSobel", ctx.warpedImageSobelBin),
            ("4unionSobel", ctx.unionSobelBinMasked),
            ("5intersectSobel", ctx.intersectSobelBinMasked),
            ("6diffMask", ctx.diffMask),
            ("7inpaintMask", ctx.inpaintMask),
            ("8inpaint", ctx.warpedImageInpaint),
            ("9ocrMask", ctx.inpaintOcrMask),
            ("10iouMask", ctx.iouMask),
            ("11inpaintProb", ctx.inpaintProbInt8),
        ])
        return ctx.finalDecision
    
    def aggregateFeatures(self, features: typing.List[typing.Any]) -> typing.Any:
        # Return the last feature
        # For typewriter animation, the last frame is the one that contains the most text 
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
        while imgH // scaleDown > self.maxDetectionSide or imgW // scaleDown > self.maxDetectionSide:
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
            if np.abs(angle) > self.maxTextAngleRad:
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
        for box in boxes:
            x, y, w, h = box.x, box.y, box.w, box.h
            if w * h <= self.nonMajorBoxSuppressionMaxRatio * maxTextBoxArea:
                break
            maxTextBoxRawHeight = max(maxTextBoxRawHeight, box.rawH)
            mask[y:y+h, x:x+w] = 255

        dialogVal: float = np.mean(mask)

        return mask, dialogVal, mask, maxTextBoxArea, maxTextBoxRawHeight
