from __future__ import annotations
import dataclasses
import os
import pytesseract
import paddleocr
import typing

import scipy.cluster.hierarchy
import scipy.spatial.distance
import sklearn.preprocessing

from IR import IIR
from Util import *
from AbstractFlagIndex import *
from IR import *
from Strategies.AbstractStrategy import *


class IIROcrPass(IIRPass):
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

            if self.doTeseract:
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


class IIRStyleClassifyPass(IIRPass):
    @dataclasses.dataclass
    class RadialProfile:
        """Per-interval colour profile produced by the radial soft-layer pipeline."""
        labMeans:       np.ndarray                       # (8, 3) float32 - weighted LAB mean per radial bin
        cohesions:      np.ndarray                       # (8,)   float32 - colour uniformity per bin
        supports:       np.ndarray                       # (8,)   float32 - relative pixel mass per bin
        localQualities: np.ndarray                       # (8,)   float32 - support × cohesion
        peerSupports:   typing.Optional[np.ndarray] = None   # (8,) float32 - set by applyPeerReinforcement
        finalWeights:   typing.Optional[np.ndarray] = None   # (8,) float32 - set by applyPeerReinforcement

    def __init__(self, config: dict, frameKey: type):
        self.frameKey: type = frameKey
        self.nonMajorBoxSuppressionMaxRatio: float = config["nonMajorBoxSuppressionMaxRatio"]
        self.nonMajorBoxSuppressionMinRank: int = config["nonMajorBoxSuppressionMinRank"]

        # extractColourClusters / runSobelCcFilter parameters
        self.sobelThreshold: int = config["sobelThreshold"]
        self.minCcAreaRatio: float = config["minCcAreaRatio"]
        self.maxCcAreaRatio: float = config["maxCcAreaRatio"]
        self.maxCcStddev: float = config["maxCcStddev"]
        self.clusterThreshold: float = config["clusterThreshold"]

        # radialSoft feature parameters
        self.radialDecayRatio: float = config["radialDecayRatio"]
        self.cohesionSigma: float = config["cohesionSigma"]
        self.peerSigma: float = config["peerSigma"]
        self.rareStyleFloor: float = config["rareStyleFloor"]
        self.weightFeatureScale: float = config["weightFeatureScale"]

        # Post-clustering rep-colour selection parameters
        self.intraSigma: float = config["intraSigma"]
        self.crossSigma: float = config["crossSigma"]

        # Inter-interval clustering parameters
        self.clusterDistThreshold: float = config["clusterDistThreshold"]
        self.minIntervalCount: int = config["minIntervalCount"]

        # Output parameters
        self.styleNames: typing.List[str] = config.get("styleNames", [])
        self.baseStyleTemplate: str = config.get(
            "baseStyleTemplate",
            "Style: {name},Microsoft YaHei,80,{primaryColour},&H000000FF,"
            "{outlineColour},&H00000000,0,0,0,0,100,100,0,0,1,4,2,2,10,10,200,1"
        )

        suppressPaddleWarnings()
        self.detector = paddleocr.TextDetection(
            model_name="PP-OCRv4_mobile_det",
            model_dir="./PaddleOCRModels/official_models/PP-OCRv4_mobile_det",
            thresh=0.2,
            box_thresh=0.3,
            device="cpu",
            enable_mkldnn=True
        )

    # ------------------------------------------------------------------
    # Private static helpers (style-classification only)
    # ------------------------------------------------------------------

    @staticmethod
    def runSobelCcFilter(
        roi: cv.Mat,
        sobelThreshold: int,
        minCcAreaRatio: float,
        maxCcAreaRatio: float,
        maxCcStddev: float,
    ) -> typing.Tuple[typing.List[int], np.ndarray, np.ndarray, typing.List[np.ndarray], int]:
        """Shared Sobel/CC filter pipeline: find flat-colour regions that don't touch the ROI border.

        Runs Sobel edge detection, inverts to get flat-colour (low-gradient) regions as connected
        components, then filters by area, colour consistency (stddev), and border non-contact.
        These accepted CCs correspond to uniform text fill pixels.

        Returns (acceptedIds, labels, stats, ccMeans, acceptedArea).
        acceptedIds is empty when no valid regions are found."""
        imageSobel = rgbSobel(roi, 1)
        imageSobelBin = cv.threshold(imageSobel, sobelThreshold, 255, cv.THRESH_BINARY_INV)[1]

        roiH, roiW = roi.shape[:2]
        area = roiH * roiW
        minCcArea = max(minCcAreaRatio * area, 10)
        maxCcArea = maxCcAreaRatio * area

        nLabels, labels, stats, _ = cv.connectedComponentsWithStats(imageSobelBin, connectivity=4, ltype=cv.CV_32S)

        acceptedIds: typing.List[int] = []
        ccMeans: typing.List[np.ndarray] = []
        acceptedArea = 0

        for i in range(nLabels):
            ccArea = stats[i][cv.CC_STAT_AREA]
            if minCcArea <= ccArea <= maxCcArea:
                mask = np.where(labels == i, 255, 0).astype(np.uint8)
                mean, std = cv.meanStdDev(roi, mask=mask)
                if float(np.mean(std)) < maxCcStddev:
                    ccLeft   = stats[i][cv.CC_STAT_LEFT]
                    ccTop    = stats[i][cv.CC_STAT_TOP]
                    ccRight  = ccLeft + stats[i][cv.CC_STAT_WIDTH]
                    ccBottom = ccTop  + stats[i][cv.CC_STAT_HEIGHT]
                    if ccLeft == 0 or ccTop == 0 or ccRight >= roiW or ccBottom >= roiH:
                        continue
                    acceptedIds.append(i)
                    ccMeans.append(mean)
                    acceptedArea += ccArea

        # Post-dilation: expand each accepted CC by 1 pixel (3×3 star/cross kernel) to compensate
        # for the Sobel edge detection eating into thin letter strokes.  Expansion is blocked by
        # pixels already owned by other accepted CCs; largest-area CCs get priority in contested
        # pixels so that large fill letters are not shrunk by smaller outline rings claiming first.
        if acceptedIds:
            starKernel = cv.getStructuringElement(cv.MORPH_CROSS, (3, 3))

            # Process in descending area order so larger (fill) CCs claim contested pixels first.
            byArea = sorted(range(len(acceptedIds)), key=lambda k: -stats[acceptedIds[k]][cv.CC_STAT_AREA])

            # Initial claimed mask: all accepted CC pixels before any dilation.
            claimedMask = np.zeros((roiH, roiW), dtype=np.uint8)
            for ccId in acceptedIds:
                claimedMask[labels == ccId] = 255

            newCcMeans: typing.List[np.ndarray] = list(ccMeans)  # filled in-order below
            newAcceptedArea = 0

            for k in byArea:
                ccId = acceptedIds[k]
                origMask = (labels == ccId).astype(np.uint8) * 255
                dilated  = cv.dilate(origMask, starKernel)

                # Expansion is allowed into pixels not claimed by OTHER accepted CCs.
                claimedByOthers = claimedMask & ~origMask
                expand = ((dilated > 0) & (claimedByOthers == 0)).astype(np.uint8) * 255

                # Claim the expanded pixels.
                labels[expand > 0] = ccId
                claimedMask[expand > 0] = 255

                # Update bbox and area stats from the expanded mask.
                ys, xs = np.where(expand > 0)
                stats[ccId][cv.CC_STAT_LEFT]   = int(xs.min())
                stats[ccId][cv.CC_STAT_TOP]    = int(ys.min())
                stats[ccId][cv.CC_STAT_WIDTH]  = int(xs.max() - xs.min() + 1)
                stats[ccId][cv.CC_STAT_HEIGHT] = int(ys.max() - ys.min() + 1)
                stats[ccId][cv.CC_STAT_AREA]   = int((expand > 0).sum())

                # Recompute mean colour from original roi pixels within expanded region.
                newMean, _ = cv.meanStdDev(roi, mask=expand)
                newCcMeans[k] = newMean
                newAcceptedArea += stats[ccId][cv.CC_STAT_AREA]

            ccMeans = newCcMeans
            acceptedArea = newAcceptedArea

        return acceptedIds, labels, stats, ccMeans, acceptedArea

    @staticmethod
    def extractColourClusters(
        roi: cv.Mat,
        sobelThreshold: int,
        minCcAreaRatio: float,
        maxCcAreaRatio: float,
        maxCcStddev: float,
        clusterThreshold: float,
        debugRoiOut: typing.Optional[cv.Mat] = None,
        outTopClusterMask: typing.Optional[np.ndarray] = None,
    ) -> typing.List[typing.Tuple[np.ndarray, float]]:
        """Extract colour clusters from a text box ROI.
        Returns [(avgColourBGR, areaRatio), ...] sorted descending by nesting score
        (sum of area x nestDepth for CCs in that cluster).  CCs whose bounding boxes are
        enclosed by other CCs' bounding boxes receive higher depth values, so core fill
        pixels (nested inside outline/shadow layers) rank above outer-ring CCs regardless
        of absolute area or compactness.
        avgColourBGR is a shape (3,) float array;
        areaRatio is the cluster's share of total accepted CC area (0.0-1.0).
        Returns empty list if no valid regions are found.
        If debugRoiOut is provided (a writable view into a debug canvas), accepted CC
        pixels are painted onto it with their original colours from roi.
        If outTopClusterMask is provided (a zeroed uint8 array of the same HxW as roi),
        pixels belonging to the top-ranked cluster are set to 255 in it."""
        acceptedIds, labels, stats, ccMeans, acceptedArea = IIRStyleClassifyPass.runSobelCcFilter(
            roi, sobelThreshold, minCcAreaRatio, maxCcAreaRatio, maxCcStddev
        )

        if debugRoiOut is not None:
            for i in acceptedIds:
                pixelMask = labels == i
                debugRoiOut[pixelMask] = roi[pixelMask]

        if len(acceptedIds) == 0:
            return []

        if len(acceptedIds) == 1:
            if outTopClusterMask is not None:
                outTopClusterMask[labels == acceptedIds[0]] = 255
            return [(ccMeans[0].flatten(), 1.0)]

        ccMeansArr = np.array(ccMeans).reshape(len(ccMeans), -1)
        ccAreasArr = np.array([stats[i][cv.CC_STAT_AREA] for i in acceptedIds], dtype=np.float64)

        # Compute nesting depth for each accepted CC using bounding-box containment.
        # A CC's direct parent is the accepted CC with the smallest bbox area that fully
        # encloses it (i.e. the tightest wrapper).  If none exists, its parent is the
        # virtual root at depth 0, so the CC itself gets depth 1.
        # Deeper CCs (fill inside outline inside shadow) receive higher multipliers,
        # which is what we want: core fill pixels rank above outer-shell pixels.
        n = len(acceptedIds)
        ccBoxes = [
            (stats[acceptedIds[i]][cv.CC_STAT_LEFT],
             stats[acceptedIds[i]][cv.CC_STAT_TOP],
             stats[acceptedIds[i]][cv.CC_STAT_LEFT] + stats[acceptedIds[i]][cv.CC_STAT_WIDTH],
             stats[acceptedIds[i]][cv.CC_STAT_TOP]  + stats[acceptedIds[i]][cv.CC_STAT_HEIGHT])
            for i in range(n)
        ]
        ccBboxAreas = [float(ccBoxes[i][2] - ccBoxes[i][0]) * float(ccBoxes[i][3] - ccBoxes[i][1])
                       for i in range(n)]

        def bbContains(outer: int, inner: int) -> bool:
            """True when outer's bbox fully contains inner's bbox (outer ≠ inner)."""
            ol, ot, or_, ob = ccBoxes[outer]
            il, it, ir, ib = ccBoxes[inner]
            return ol <= il and ir <= or_ and ot <= it and ib <= ob

        # directParent[i] = index into acceptedIds of i's tightest enclosing CC, or -1.
        directParent: typing.List[int] = []
        for i in range(n):
            bestArea = float('inf')
            parent = -1
            for j in range(n):
                if j != i and bbContains(j, i) and ccBboxAreas[j] < bestArea:
                    bestArea = ccBboxAreas[j]
                    parent = j
            directParent.append(parent)

        # Memoised depth: depth[i] = depth[directParent[i]] + 1; virtual root = 0.
        ccNestDepths: typing.List[int] = [0] * n

        def getDepth(i: int) -> int:
            if ccNestDepths[i] != 0:
                return ccNestDepths[i]
            d = 1 if directParent[i] == -1 else getDepth(directParent[i]) + 1
            ccNestDepths[i] = d
            return d

        for i in range(n):
            getDepth(i)

        scaler = sklearn.preprocessing.StandardScaler()
        ccMeansScaled = scaler.fit_transform(ccMeansArr)

        distMat = scipy.spatial.distance.pdist(ccMeansScaled, metric='euclidean')
        Z = scipy.cluster.hierarchy.linkage(distMat, method='weighted')
        clusterAssign = scipy.cluster.hierarchy.fcluster(Z, clusterThreshold, criterion='distance')

        clusterColours: typing.Dict[int, typing.List[typing.Tuple[np.ndarray, float]]] = {}
        clusterAreas: typing.Dict[int, float] = {}
        clusterNestScores: typing.Dict[int, float] = {}
        clusterCcIds: typing.Dict[int, typing.List[int]] = {}

        for idx, (ccId, cid) in enumerate(zip(acceptedIds, clusterAssign)):
            cid = int(cid)
            if cid not in clusterColours:
                clusterColours[cid] = []
                clusterAreas[cid] = 0.0
                clusterNestScores[cid] = 0.0
                clusterCcIds[cid] = []
            clusterColours[cid].append((ccMeansArr[idx], ccAreasArr[idx]))
            clusterAreas[cid] += ccAreasArr[idx]
            clusterNestScores[cid] += ccAreasArr[idx] * ccNestDepths[idx]
            clusterCcIds[cid].append(ccId)

        # Sort cluster IDs by nesting score descending (deeper fill > outer shell).
        sortedCids = sorted(clusterNestScores, key=lambda c: clusterNestScores[c], reverse=True)

        # Fill the top-cluster mask with an outer-border pruning step.
        #
        # Only "root" CCs within the top cluster (those not bbox-contained by any other
        # top-cluster CC) are candidates for removal.  CCs that are already inside another
        # same-cluster CC are permanently safe.  This is a single-pass, one-layer-only
        # prune: no cascading, once a root CC is removed its former children become safe
        # automatically because they now have no same-cluster parent.
        #
        # Removal criterion for a root CC that contains other same-cluster CCs:
        #   FR_self  < 0.5 × FR_subtree
        # where FR_subtree is the area-weighted mean fill ratio of all same-cluster CCs
        # contained within this CC's bbox.  A thin outer border has a low fill ratio
        # relative to the compact fill letters it surrounds, so it is pruned; genuine
        # character strokes have similar fill ratios to the
        # strokes they enclose and are kept.
        if outTopClusterMask is not None and sortedCids:
            ccIdToIdx = {acceptedIds[i]: i for i in range(n)}
            topIndices = [ccIdToIdx[ccId] for ccId in clusterCcIds[sortedCids[0]]]

            # containedBy[i] = all top-cluster indices j whose bbox lies inside bbox_i.
            containedBy: typing.Dict[int, typing.List[int]] = {i: [] for i in topIndices}
            hasParentInCluster: typing.Set[int] = set()
            for i in topIndices:
                for j in topIndices:
                    if i != j and bbContains(i, j):
                        containedBy[i].append(j)
                        hasParentInCluster.add(j)

            kept: typing.Set[int] = set(topIndices)
            for i in topIndices:
                if i in hasParentInCluster:
                    continue  # safe: already inside another same-cluster CC
                children = containedBy[i]
                if not children:
                    continue  # leaf root: safe
                # Root CC that wraps other same-cluster CCs: apply FR criterion.
                frSelf = float(stats[acceptedIds[i]][cv.CC_STAT_AREA]) / max(
                    float(stats[acceptedIds[i]][cv.CC_STAT_WIDTH]) *
                    float(stats[acceptedIds[i]][cv.CC_STAT_HEIGHT]), 1.0
                )
                totalChildArea = sum(float(stats[acceptedIds[j]][cv.CC_STAT_AREA]) for j in children)
                frSubtree = (
                    sum(
                        float(stats[acceptedIds[j]][cv.CC_STAT_AREA]) *
                        float(stats[acceptedIds[j]][cv.CC_STAT_AREA]) / max(
                            float(stats[acceptedIds[j]][cv.CC_STAT_WIDTH]) *
                            float(stats[acceptedIds[j]][cv.CC_STAT_HEIGHT]), 1.0
                        )
                        for j in children
                    ) / totalChildArea
                    if totalChildArea > 0 else 0.0
                )
                if frSelf < 0.5 * frSubtree:
                    kept.discard(i)

            for i in kept:
                outTopClusterMask[labels == acceptedIds[i]] = 255

        result: typing.List[typing.Tuple[np.ndarray, float]] = []
        for cid in sortedCids:
            totalArea = clusterAreas[cid]
            areaRatio = totalArea / acceptedArea
            weightedSum = np.sum([mean * a for mean, a in clusterColours[cid]], axis=0)
            result.append((weightedSum / totalArea, areaRatio))

        return result

    @staticmethod
    def extractLocalProfile(
        roi: cv.Mat,
        sobelThreshold: int,
        minCcAreaRatio: float,
        maxCcAreaRatio: float,
        maxCcStddev: float,
        radialDecayRatio: float,
        cohesionSigma: float,
        clusterThreshold: float,
        debugRoiOut: typing.Optional[cv.Mat] = None,
    ) -> typing.Optional[IIRStyleClassifyPass.RadialProfile]:
        """Extract radial soft-layer profile from a text box ROI (Stage A of radialSoft algorithm).

        Uses extractColourClusters to colour-cluster accepted CCs sorted by fill score
        (sum of area x fillRatio**2).  The top-ranked cluster - compact fill letters - becomes
        the coreMask origin for the radial distance transform.  Thin-ring outline CCs have
        lower fillRatio and rank below the fill cluster regardless of absolute area.

        If debugRoiOut is provided (writable view into a debug canvas), core pixels are painted onto it.

        Returns None if the Sobel/CC pipeline finds no core area."""
        roiH, roiW = roi.shape[:2]

        coreMask = np.zeros((roiH, roiW), dtype=np.uint8)
        clusters = IIRStyleClassifyPass.extractColourClusters(
            roi, sobelThreshold, minCcAreaRatio, maxCcAreaRatio, maxCcStddev,
            clusterThreshold,
            outTopClusterMask=coreMask,
        )

        if not clusters or not coreMask.any():
            return None

        if debugRoiOut is not None:
            coreMaskBool = coreMask.astype(bool)
            debugRoiOut[coreMaskBool] = roi[coreMaskBool]

        # Distance from every pixel to the nearest core pixel.
        # outsideMask is 1 (non-zero) outside the core; distanceTransform measures distance
        # of each non-zero pixel to the nearest zero pixel (= nearest core pixel).
        # Core pixels themselves are zero in outsideMask, so their output distance is 0.
        outsideMask = (coreMask == 0).astype(np.uint8)
        distMap = cv.distanceTransform(outsideMask, cv.DIST_L2, 3).astype(np.float32)

        baseUnit = float(min(roiW, roiH))
        decayRadius = max(1.0, radialDecayRatio * baseUnit)
        normDist = np.minimum(distMap / decayRadius, 1.0).astype(np.float32)
        decayWeight = (1.0 - normDist).astype(np.float32)
        totalDecayMass = float(decayWeight.sum())

        cropLab = cv.cvtColor(roi, cv.COLOR_BGR2LAB).astype(np.float32)

        binCount = 8
        binCenters = np.linspace(0.0, 1.0, binCount, dtype=np.float32)
        binHalfWidth = float(1.0 / (binCount - 1))
        eps = 1e-6

        labMeans = np.zeros((binCount, 3), dtype=np.float32)
        cohesions = np.zeros(binCount, dtype=np.float32)
        supports = np.zeros(binCount, dtype=np.float32)
        localQualities = np.zeros(binCount, dtype=np.float32)

        # Smaller exponent -> outer bins get relatively more influence; larger → core dominates.
        SUPPORT_EXPONENT = 0.75
        for k in range(binCount):
            center = float(binCenters[k])
            kernel = np.maximum(0.0, 1.0 - np.abs(normDist - center) / binHalfWidth).astype(np.float32)
            pixelWeight = decayWeight * kernel  # (H, W)
            weightSum = float(pixelWeight.sum())
            if weightSum < eps:
                continue  # leave bin as zeros

            labMean = (cropLab * pixelWeight[..., None]).sum(axis=(0, 1)) / weightSum  # (3,)
            labDiff = cropLab - labMean[None, None, :]  # (H, W, 3)
            labSqDist = (labDiff * labDiff).sum(axis=2)  # (H, W)
            labVar = float((labSqDist * pixelWeight).sum() / weightSum)

            support = weightSum / (totalDecayMass + eps)
            # Gate cohesion: low-support bins have too few samples for a reliable variance
            # estimate and can spuriously show near-1 cohesion.  Only compute it when the
            # bin has meaningful mass; otherwise leave cohesion at 0.
            if support >= 0.05:
                cohesion = float(np.exp(-labVar / (cohesionSigma * cohesionSigma)))
            else:
                cohesion = 0.0

            labMeans[k] = labMean
            cohesions[k] = cohesion
            supports[k] = support
            localQualities[k] = float(support ** SUPPORT_EXPONENT)

        # B0 (core fill) cohesion is forced to 1.0: the core mask captures the dominant fill
        # colour, so any variance at B0 comes from legitimate fill-colour diversity (e.g. when
        # coreMask still includes some outline pixels), not from noise.  Penalising it with a
        # low cohesion would unjustly block fill colours from rep-colour candidacy.
        cohesions[0] = 1.0

        return IIRStyleClassifyPass.RadialProfile(
            labMeans=labMeans,
            cohesions=cohesions,
            supports=supports,
            localQualities=localQualities,
        )

    @staticmethod
    def mergeBoxProfiles(
        boxProfiles: typing.List[typing.Tuple[IIRStyleClassifyPass.RadialProfile, float]]
    ) -> IIRStyleClassifyPass.RadialProfile:
        """Merge multiple per-box local profiles into a single interval profile by area-weighted averaging."""
        totalArea = sum(a for _, a in boxProfiles)
        def wavg(attr: str) -> np.ndarray:
            acc = np.zeros_like(getattr(boxProfiles[0][0], attr))
            for profile, area in boxProfiles:
                acc = acc + getattr(profile, attr) * (area / totalArea)
            return acc
        return IIRStyleClassifyPass.RadialProfile(
            labMeans=wavg("labMeans"),
            cohesions=wavg("cohesions"),
            supports=wavg("supports"),
            localQualities=wavg("localQualities"),
        )

    @staticmethod
    def applyPeerReinforcement(
        allProfiles: typing.List[typing.Optional[IIRStyleClassifyPass.RadialProfile]],
        peerSigma: float,
        rareStyleFloor: float,
    ) -> None:
        """Compute peer support and final weights for all profiles in-place (Stage B).

        For each sample i and radial bin k, peerSupport measures how many other samples
        in the same bin share a similar colour with high local quality. This amplifies
        colours that recur consistently across the video (style signals) and suppresses
        one-off colours that are likely background contamination.

        Mutates each non-None profile by adding 'peerSupports' and 'finalWeights' keys."""
        eps = 1e-6
        validIndices = [i for i, p in enumerate(allProfiles) if p is not None]
        N = len(validIndices)

        if N == 0:
            return

        binCount = 8
        validProfiles: typing.List[IIRStyleClassifyPass.RadialProfile] = [
            typing.cast(IIRStyleClassifyPass.RadialProfile, allProfiles[i]) for i in validIndices
        ]
        allLabMeans = np.stack([p.labMeans for p in validProfiles]).astype(np.float32)      # (N, 8, 3)
        allLocalQuality = np.stack([p.localQualities for p in validProfiles]).astype(np.float32)  # (N, 8)

        allLabMeansT = allLabMeans.transpose(1, 0, 2)  # (8, N, 3)
        allQualityT = allLocalQuality.T                  # (8, N)
        qualityRowSums = allQualityT.sum(axis=1)         # (8,) - denominator base, same for all ni

        twoSigmaSq = float(2.0 * peerSigma * peerSigma)
        peerSupportsAll = np.zeros((N, binCount), dtype=np.float32)

        for ni in range(N):
            diff = allLabMeansT - allLabMeans[ni][:, None, :]  # (8, N, 3)
            distSq = (diff * diff).sum(axis=2)                  # (8, N)
            colorSim = np.exp(-distSq / twoSigmaSq)             # (8, N)
            selfQuality = allQualityT[:, ni]                                          # (8,)
            numerator = (allQualityT * colorSim).sum(axis=1) - selfQuality           # (8,)
            denom = qualityRowSums - selfQuality + eps                                # (8,)
            peerSupportsAll[ni] = numerator / denom

        for n, p in enumerate(validProfiles):
            peerSupports = peerSupportsAll[n]
            finalWeights = p.localQualities * (rareStyleFloor + (1.0 - rareStyleFloor) * peerSupports)
            p.peerSupports = peerSupports.astype(np.float32)
            p.finalWeights = finalWeights.astype(np.float32)

    @staticmethod
    def buildStyleOutputs(
        profile: IIRStyleClassifyPass.RadialProfile,
        weightFeatureScale: float,
    ) -> np.ndarray:
        """Build 32-dim style feature vector from a profile (for agglomerative clustering).

        Returns:
            styleFeatureVector: np.ndarray shape (32,), float32"""
        assert profile.finalWeights is not None, "buildStyleOutputs requires peer reinforcement to have run"
        finalWeights = profile.finalWeights  # (8,)
        labMeans = profile.labMeans          # (8, 3)

        # Normalise weights to relative proportions so the feature vector captures the
        # colour *distribution* rather than absolute mass.  Two intervals with the same
        # style but different text density will then have similar feature vectors.
        totalFW = float(finalWeights.sum())
        normFW = finalWeights / (totalFW + 1e-6)

        binFeatures = np.empty((8, 4), dtype=np.float32)
        for k in range(8):
            w = float(normFW[k])
            lm = labMeans[k]
            lNorm = lm[0] / 255.0
            aNorm = (lm[1] - 128.0) / 128.0
            bNorm = (lm[2] - 128.0) / 128.0
            wSqrt = float(np.sqrt(max(w, 0.0)))
            binFeatures[k] = [wSqrt * lNorm, wSqrt * aNorm, wSqrt * bNorm, weightFeatureScale * w]
        return binFeatures.flatten()

    @staticmethod
    def selectClusterRepColour(
        memberProfiles: typing.List['IIRStyleClassifyPass.RadialProfile'],
        allClusterMeans: np.ndarray,
        thisClusterIdx: int,
        intraSigma: float,
        crossSigma: float,
        debug: bool = False,
    ) -> typing.Tuple[np.ndarray, np.ndarray]:
        """Select the most representative colour bin for a style cluster (post-clustering Stage D).

        For each radial bin k, two soft scores are computed:

          intraBinding[k]:   how consistently all member profiles share the same colour at bin k.
                             Gaussian decay of the support-weighted mean intra-cluster LAB spread.
                             1 = all frames agree, 0 = chaotic across frames.

          externContrast[k]: how distinct this cluster's bin-k colour is from all other clusters.
                             1 - Gaussian decay of the minimum cross-cluster LAB distance.
                             1 = unique to this style, 0 = identical to another cluster.

          repScore[k] = intraBinding[k] x externContrast[k]

        The bin with the highest repScore supplies the representative colour.
        Ties (e.g. all zeros when profiles are empty) fall back to bin 0 (core fill).

        Args:
            memberProfiles:   profiles of all valid member intervals in this cluster.
            allClusterMeans:  (numClusters, 8, 3) float32 - support-weighted LAB mean per bin per cluster,
                              pre-computed by the caller so all clusters share a consistent reference.
            thisClusterIdx:   index of this cluster in allClusterMeans.
            intraSigma:       LAB-space sigma for intraBinding Gaussian (higher = more tolerant of spread).
            crossSigma:       LAB-space sigma for externContrast Gaussian (higher = need greater distance to contrast).

        Returns:
            BGR uint8 array shape (3,) - the representative colour."""
        binCount = 8
        eps = 1e-6
        twoIntraSq = 2.0 * intraSigma * intraSigma + eps
        twoCrossSq = 2.0 * crossSigma * crossSigma + eps

        clusterMean = allClusterMeans[thisClusterIdx]  # (8, 3)

        # intraBinding: support-weighted mean squared LAB distance from each member to cluster mean.
        # Single-member clusters get intraBinding = 1.0 (no spread to measure).
        intraBinding = np.ones(binCount, dtype=np.float32)
        if len(memberProfiles) > 1:
            for k in range(binCount):
                totalW = 0.0
                weightedDistSq = 0.0
                for p in memberProfiles:
                    w = float(p.supports[k])
                    diff = p.labMeans[k] - clusterMean[k]
                    weightedDistSq += w * float(np.dot(diff, diff))
                    totalW += w
                meanDistSq = weightedDistSq / totalW if totalW > eps else 0.0
                intraBinding[k] = float(np.exp(-meanDistSq / twoIntraSq))

        # externContrast: 1 - Gaussian decay of min LAB distance to any other cluster's mean.
        # Single cluster (no others) gets externContrast = 1.0 everywhere.
        numClusters = allClusterMeans.shape[0]
        externContrast = np.ones(binCount, dtype=np.float32)
        otherIndices = [c for c in range(numClusters) if c != thisClusterIdx]
        if otherIndices:
            for k in range(binCount):
                minDistSq = float('inf')
                for c in otherIndices:
                    diff = clusterMean[k] - allClusterMeans[c, k]
                    distSq = float(np.dot(diff, diff))
                    if distSq < minDistSq:
                        minDistSq = distSq
                externContrast[k] = 1.0 - float(np.exp(-minDistSq / twoCrossSq))

        # B0 is always the primary (inner fill) colour — exclude it from outline competition.
        repScore = intraBinding * externContrast  # (8,)
        outlineScore = repScore[1:]               # B1-B7 only
        bestOutlineBin = int(np.argmax(outlineScore)) + 1

        if debug:
            fmt = lambda arr: "".join(f"{float(x):6.2f}" for x in arr)
            hdr = "".join(f"{'B'+str(k):>6}" for k in range(8))
            print(f"    [sty] cluster {thisClusterIdx}:")
            print(f"          bins : {hdr}")
            print(f"          intra: {fmt(intraBinding)}")
            print(f"          ext  : {fmt(externContrast)}")
            print(f"          rep  : {fmt(repScore)}  -> outline B{bestOutlineBin}")

        def labToBgr(lab: np.ndarray) -> np.ndarray:
            px = np.clip(lab, 0, 255).reshape(1, 1, 3).astype(np.uint8)
            return cv.cvtColor(px, cv.COLOR_LAB2BGR)[0, 0].astype(np.uint8)

        primaryBgr = labToBgr(clusterMean[0])
        outlineBgr = labToBgr(clusterMean[bestOutlineBin])
        return primaryBgr, outlineBgr

    @staticmethod
    def makeRadialSoftDebugImage(
        profile: IIRStyleClassifyPass.RadialProfile,
        repColoursBgr: typing.List[np.ndarray],
        repWeights: typing.List[float],
    ) -> np.ndarray:
        """Build a compact per-interval debug strip showing radialSoft bin statistics.

        Layout (left-label column + 8 bin columns):
          Row 0 - column headers  (B0 … B7)
          Row 1 - LAB mean colour swatches
          Row 2 - support    (greyscale brightness + numeric overlay)
          Row 3 - cohesion   (greyscale brightness + numeric overlay)
          Row 4 - peerSpt    (greyscale brightness + numeric overlay)
          Row 5 - finalWt    (greyscale brightness + numeric overlay)
          Row 6 - representative colours proportional to weight
        """
        LABEL_W  = 72
        CELL_W   = 52
        H_HDR    = 18
        H_COLOUR = 48
        H_SCALAR = 28
        H_REP    = 40
        N_BINS   = 8

        SCALAR_ROWS = [
            ("support",  "supports"),
            ("cohesion", "cohesions"),
            ("peerSpt",  "peerSupports"),
            ("finalWt",  "finalWeights"),
        ]

        totalW = LABEL_W + N_BINS * CELL_W
        totalH = H_HDR + H_COLOUR + len(SCALAR_ROWS) * H_SCALAR + 2 + H_REP
        img = np.full((totalH, totalW, 3), 40, dtype=np.uint8)

        font = cv.FONT_HERSHEY_SIMPLEX
        fs   = 0.38
        th   = 1

        # Row 0: column headers
        y0 = 0
        cv.putText(img, "bin", (4, y0 + H_HDR - 4), font, fs, (160, 160, 160), th, cv.LINE_AA)
        for k in range(N_BINS):
            x = LABEL_W + k * CELL_W
            cv.rectangle(img, (x, y0), (x + CELL_W - 1, y0 + H_HDR - 1), (55, 55, 55), -1)
            cv.putText(img, f"B{k}", (x + 4, y0 + H_HDR - 4), font, fs, (200, 200, 200), th, cv.LINE_AA)

        # Row 1: LAB mean colour swatches
        y0 += H_HDR
        cv.putText(img, "colour", (4, y0 + H_COLOUR // 2 + 5), font, fs, (160, 160, 160), th, cv.LINE_AA)
        for k in range(N_BINS):
            x = LABEL_W + k * CELL_W
            labPx = np.clip(profile.labMeans[k], 0, 255).reshape(1, 1, 3).astype(np.uint8)
            bgr = cv.cvtColor(labPx, cv.COLOR_LAB2BGR)[0, 0]
            cv.rectangle(img, (x, y0), (x + CELL_W - 1, y0 + H_COLOUR - 1),
                         (int(bgr[0]), int(bgr[1]), int(bgr[2])), -1)

        # Rows 2-5: scalar rows (greyscale brightness = value; auto-contrast text)
        y0 += H_COLOUR
        for rowLabel, arrKey in SCALAR_ROWS:
            raw = getattr(profile, arrKey)
            arr: np.ndarray = raw if raw is not None else np.zeros(N_BINS, dtype=np.float32)
            cv.putText(img, rowLabel, (4, y0 + H_SCALAR - 6), font, fs, (160, 160, 160), th, cv.LINE_AA)
            for k in range(N_BINS):
                x = LABEL_W + k * CELL_W
                val = float(np.clip(arr[k], 0.0, 1.0))
                grey = int(val * 255)
                cv.rectangle(img, (x, y0), (x + CELL_W - 1, y0 + H_SCALAR - 1),
                             (grey, grey, grey), -1)
                txtCol = (0, 0, 0) if grey >= 128 else (255, 255, 255)
                cv.putText(img, f"{val:.2f}", (x + 4, y0 + H_SCALAR - 7),
                           font, fs, txtCol, th, cv.LINE_AA)
            y0 += H_SCALAR

        # Divider
        cv.rectangle(img, (0, y0), (totalW - 1, y0 + 1), (80, 80, 80), -1)
        y0 += 2

        # Row 6: representative colours proportional to weight
        cv.putText(img, "repClr", (4, y0 + H_REP - 6), font, fs, (160, 160, 160), th, cv.LINE_AA)
        barW = N_BINS * CELL_W
        if repColoursBgr and repWeights:
            xCursor = LABEL_W
            totalRw = sum(repWeights) or 1.0
            for colour, weight in zip(repColoursBgr, repWeights):
                segW = max(1, round((weight / totalRw) * barW))
                segW = min(segW, LABEL_W + barW - xCursor)
                c = np.clip(colour, 0, 255).astype(np.uint8)
                cv.rectangle(img, (xCursor, y0), (xCursor + segW - 1, y0 + H_REP - 1),
                             (int(c[0]), int(c[1]), int(c[2])), -1)
                xCursor += segW
        else:
            cv.rectangle(img, (LABEL_W, y0), (LABEL_W + barW - 1, y0 + H_REP - 1), (50, 50, 50), -1)
            cv.putText(img, "none", (LABEL_W + 4, y0 + H_REP - 6),
                       font, fs, (120, 120, 120), th, cv.LINE_AA)

        return img

    # ------------------------------------------------------------------

    def detectBoxes(self, image: np.ndarray) -> typing.List[typing.Tuple[int, int, int, int]]:
        """Run text detection on image and return non-major-box-suppressed axis-aligned boxes."""
        imgH, imgW = image.shape[:2]
        result = self.detector.predict(image)[0]
        dtPolys: typing.List[np.ndarray] = result["dt_polys"]

        rawBoxes = []
        for poly in dtPolys:
            poly = np.array(poly, np.int32)
            x0, y0 = poly[0]; x1, y1 = poly[1]; x2, y2 = poly[2]; x3, y3 = poly[3]
            angle = (np.arctan2(y1 - y0, x1 - x0) + np.arctan2(y2 - y3, x2 - x3)) / 2
            if np.abs(angle) > np.pi / 180 * 3:
                continue
            bx, by, bw, bh = cv.boundingRect(poly)
            pad = round(0.1 * min(bw, bh))
            x1 = max(0, bx - pad);      y1 = max(0, by - pad)
            x2 = min(imgW, bx + bw + pad); y2 = min(imgH, by + bh + pad)
            rawBoxes.append((x1, y1, x2 - x1, y2 - y1))

        rawBoxes.sort(key=lambda b: b[2] * b[3], reverse=True)
        boxSizeSum = sum(b[2] * b[3] for b in rawBoxes)
        filteredBoxes = []
        for rank, box in enumerate(rawBoxes):
            if box[2] * box[3] <= self.nonMajorBoxSuppressionMaxRatio * boxSizeSum and rank >= self.nonMajorBoxSuppressionMinRank:
                break
            filteredBoxes.append(box)

        return filteredBoxes

    def computeAllFeatures(self, iir: IIR) -> typing.Tuple[
        typing.List[typing.Optional[np.ndarray]],
        typing.List[typing.Optional['IIRStyleClassifyPass.RadialProfile']]
    ]:
        """Two-pass feature extraction using the radial soft-layer algorithm.

        Pass 1: for each interval, run extractLocalProfile on each detected text box crop.
                The Sobel/CC pipeline identifies core fill pixels (coreMask); radial bins
                then describe colour from core outward, capturing outline and shadow layers.
                Per-box profiles are area-weighted and merged into one profile per interval.
        Global: applyPeerReinforcement cross-correlates all interval profiles - bins that
                show a consistent colour across many intervals are boosted; one-off bins
                (likely background) are attenuated.
        Pass 2: buildStyleOutputs converts each profile into a 32-dim feature vector and
                a list of representative BGR colours."""
        debug: bool = True

        # --- Phase 1: extract local profiles ---
        allProfiles: typing.List[typing.Optional[IIRStyleClassifyPass.RadialProfile]] = []

        for i, interval in enumerate(iir.intervals):
            image: cv.Mat = interval.getAttachment(self.frameKey)

            if image is None:
                allProfiles.append(None)
                continue

            boxes = self.detectBoxes(image)
            debugCcImg: typing.Optional[cv.Mat] = \
                checkerboardBackground(image.shape[1], image.shape[0]) if debug else None

            boxProfiles: typing.List[typing.Tuple[IIRStyleClassifyPass.RadialProfile, float]] = []

            for bx, by, bw, bh in boxes:
                crop: cv.Mat = typing.cast(cv.Mat, image[by:by + bh, bx:bx + bw].copy())
                if crop.size == 0:
                    continue
                debugRoiOut: typing.Optional[cv.Mat] = \
                    typing.cast(cv.Mat, debugCcImg[by:by + bh, bx:bx + bw]) if debugCcImg is not None else None

                profile = self.extractLocalProfile(
                    crop,
                    self.sobelThreshold,
                    self.minCcAreaRatio,
                    self.maxCcAreaRatio,
                    self.maxCcStddev,
                    self.radialDecayRatio,
                    self.cohesionSigma,
                    self.clusterThreshold,
                    debugRoiOut=debugRoiOut,
                )
                if profile is not None:
                    boxProfiles.append((profile, float(bw * bh)))

            if debugCcImg is not None:
                timeStr = interval.timeStringBegin().replace(":", "-")
                os.makedirs("STY_debug", exist_ok=True)
                cv.imwrite(os.path.join("STY_debug", f"{timeStr}_full.png"), image)
                cv.imwrite(os.path.join("STY_debug", f"{timeStr}_cc.png"), debugCcImg)

            allProfiles.append(self.mergeBoxProfiles(boxProfiles) if boxProfiles else None)

            if i % 10 == 0:
                print(interval.getName(i))

        # --- Global: peer reinforcement ---
        self.applyPeerReinforcement(allProfiles, self.peerSigma, self.rareStyleFloor)

        # --- Phase 2: build feature vectors (rep-colour selection deferred to post-clustering) ---
        features: typing.List[typing.Optional[np.ndarray]] = []

        for interval, profile in zip(iir.intervals, allProfiles):
            if profile is None:
                features.append(None)
                continue

            styleVec = self.buildStyleOutputs(profile, self.weightFeatureScale)
            features.append(styleVec)

            if debug:
                debugFeatImg = IIRStyleClassifyPass.makeRadialSoftDebugImage(profile, [], [])
                timeStr = interval.timeStringBegin().replace(":", "-")
                os.makedirs("STY_debug", exist_ok=True)
                cv.imwrite(os.path.join("STY_debug", f"{timeStr}_feat.png"), debugFeatImg)

        return features, allProfiles

    def cluster(self, features: typing.List[typing.Optional[np.ndarray]], validIndices: typing.List[int]) -> typing.List[int]:
        """Cluster valid intervals by their features using euclidean distance.
        Returns cluster assignments parallel to validIndices.

        When clusterDistThreshold <= 0, uses scipy's inconsistency criterion with
        depth=2 and t=2.0: each merge node is cut if its height is more than 2 standard
        deviations above the mean height of the two levels below it.  This detects all
        class boundaries independently and works well with many clusters.
        When clusterDistThreshold > 0, falls back to a fixed distance threshold."""
        if len(validIndices) == 1:
            return [0]

        featureMat = np.array([features[i] for i in validIndices])
        distMat = scipy.spatial.distance.pdist(featureMat, metric='euclidean')
        distMat = np.nan_to_num(distMat, nan=1.0)
        Z = scipy.cluster.hierarchy.linkage(distMat, method='average')

        if self.clusterDistThreshold > 0:
            print(f"  [sty] clustering: fixed distance threshold = {self.clusterDistThreshold:.4f}")
            rawLabels = scipy.cluster.hierarchy.fcluster(Z, self.clusterDistThreshold, criterion='distance')
        else:
            print(f"  [sty] clustering: inconsistency criterion (depth=2, t=2.0)")
            rawLabels = scipy.cluster.hierarchy.fcluster(Z, 1.5, depth=2, criterion='inconsistent')

        return rawLabels.tolist()

    def apply(self, iir: IIR):
        print(f"IIRStyleClassifyPass: processing {len(iir.intervals)} intervals")
        debug: bool = True

        # 1. Compute features and per-interval profiles
        features, allProfiles = self.computeAllFeatures(iir)

        # 2. Filter intervals with valid features
        validIndices = [i for i, f in enumerate(features) if f is not None]
        print(f"IIRStyleClassifyPass: {len(validIndices)} intervals have valid colour features")

        if len(validIndices) < 2:
            print("IIRStyleClassifyPass: not enough valid intervals for clustering, skipping")
            return

        # 3. Agglomerative clustering
        clusterAssignments = self.cluster(features, validIndices)

        # 4. Reorder clusters by member count (largest first) for stable naming
        from collections import Counter
        clusterCounts = Counter(clusterAssignments)
        sizeOrder = [cid for cid, _ in clusterCounts.most_common()]
        remapping = {old: new for new, old in enumerate(sizeOrder)}
        clusterAssignments = [remapping[c] for c in clusterAssignments]

        nClusters = len(set(clusterAssignments))
        print(f"IIRStyleClassifyPass: {nClusters} clusters found")

        # 5. Pre-compute support-weighted LAB mean per bin for every cluster.
        #    Used by selectClusterRepColour for both intra-binding and extern-contrast.
        allClusterMeans = np.zeros((nClusters, 8, 3), dtype=np.float32)
        allClusterWeights = np.zeros((nClusters, 8), dtype=np.float32)
        for i, clusterId in zip(validIndices, clusterAssignments):
            p = allProfiles[i]
            if p is None:
                continue
            for k in range(8):
                w = float(p.supports[k])
                allClusterMeans[clusterId, k] += p.labMeans[k] * w
                allClusterWeights[clusterId, k] += w
        for c in range(nClusters):
            for k in range(8):
                if allClusterWeights[c, k] > 0:
                    allClusterMeans[c, k] /= allClusterWeights[c, k]

        # 6. Generate style declarations and assign styles to intervals
        clusterStyleNames: typing.Dict[int, str] = {}
        newStyleLines: typing.List[str] = []

        for clusterId in sorted(set(clusterAssignments)):
            memberIndices = [validIndices[i] for i, c in enumerate(clusterAssignments) if c == clusterId]
            if len(memberIndices) < self.minIntervalCount:
                continue

            memberProfiles = [
                typing.cast(IIRStyleClassifyPass.RadialProfile, allProfiles[i])
                for i in memberIndices if allProfiles[i] is not None
            ]
            repColour = self.selectClusterRepColour(
                memberProfiles, allClusterMeans, clusterId,
                self.intraSigma, self.crossSigma,
                debug=debug,
            )

            if self.styleNames and clusterId < len(self.styleNames):
                styleName = self.styleNames[clusterId]
            else:
                styleName = f"StyleClass_{clusterId}"

            primaryColour, outlineColour = repColour
            pb, pg, pr = primaryColour
            ob, og, or_ = outlineColour
            assPrimary = f"&H00{pb:02X}{pg:02X}{pr:02X}"
            assOutline = f"&H00{ob:02X}{og:02X}{or_:02X}"
            styleLine = self.baseStyleTemplate.format(
                name=styleName, primaryColour=assPrimary, outlineColour=assOutline
            )
            newStyleLines.append(styleLine)
            clusterStyleNames[clusterId] = styleName

        # Atomically: append style declarations AND assign interval styles
        iir.styles.extend(newStyleLines)

        for i, clusterId in zip(validIndices, clusterAssignments):
            if clusterId in clusterStyleNames:
                iir.intervals[i].style = clusterStyleNames[clusterId]

        print(f"IIRStyleClassifyPass: assigned {len(clusterStyleNames)} styles, "
              f"declared {len(newStyleLines)} style lines")
