from __future__ import annotations
import abc
import typing
import numpy as np
import cv2 as cv
import datetime
import argparse
import enum

class AbstractRect(abc.ABC):
    def getNest(self) -> typing.Tuple[int, int, float, float, float, float]:
        # returns: fullHeight, fullWidth, topOffset, leftOffset, verticalCompressRate, horizontalCompressRate
        return 0, 0, 0.0, 0.0, 0.0, 0.0

    def cutRoi(self, frame: cv.Mat) -> cv.Mat:
        return cv.Mat((1, 1))

    def getSize(self) -> typing.Tuple[int, int]:
        # returns: width, height
        # notice this order!
        return 0, 0

    def getWidth(self) -> int:
        return self.getSize()[0]

    def getHeight(self) -> int:
        return self.getSize()[1]

class RatioRect(AbstractRect):
    def __init__(self, parent: AbstractRect, topRatio: float, bottomRatio: float, leftRatio: float, rightRatio: float) -> None:
        self.parent: AbstractRect = parent
        self.topRatio: float = topRatio
        self.bottomRatio: float = bottomRatio
        self.leftRatio: float = leftRatio
        self.rightRatio: float = rightRatio

        # Precalculate offsets
        fullHeight, fullWidth, topOffset, leftOffset, verticalCompressRate, horizontalCompressRate = self.parent.getNest()
        self.fullHeight: int = fullHeight
        self.fullWidth: int = fullWidth
        self.topOffset: float = topOffset + self.fullHeight * verticalCompressRate * self.topRatio
        self.bottomOffset: float = topOffset + self.fullHeight * verticalCompressRate * self.bottomRatio
        self.leftOffset: float = leftOffset + self.fullWidth * horizontalCompressRate * self.leftRatio
        self.rightOffset: float = leftOffset + self.fullWidth * horizontalCompressRate * self.rightRatio
        self.verticalCompressRate: float = verticalCompressRate * (self.bottomRatio - self.topRatio)
        self.horizontalCompressRate: float = horizontalCompressRate * (self.rightRatio - self.leftRatio)
        self.localHeight: int = int(self.fullHeight * self.verticalCompressRate)
        self.localWidth: int = int(self.fullWidth * self.horizontalCompressRate)

    def getNest(self) -> typing.Tuple[int, int, float, float, float, float]:
        return self.fullHeight, self.fullWidth, self.topOffset, self.leftOffset, self.verticalCompressRate, self.horizontalCompressRate

    def cutRoi(self, frame: cv.Mat) -> cv.Mat:
        return frame[int(self.topOffset):int(self.bottomOffset), int(self.leftOffset):int(self.rightOffset)]

    def getSize(self) -> typing.Tuple[int, int]:
        return self.localWidth, self.localHeight # notice this order!

class SrcRect(AbstractRect):
    def __init__(self, src: cv.VideoCapture):
        self.height: int = int(src.get(cv.CAP_PROP_FRAME_HEIGHT))
        self.width: int = int(src.get(cv.CAP_PROP_FRAME_WIDTH))

    def getNest(self) -> typing.Tuple[int, int, float, float, float, float]:
        return self.height, self.width, 0.0, 0.0, 1.0, 1.0

    def cutRoi(self, frame: cv.Mat) -> cv.Mat:
        return frame

    def getSize(self) -> typing.Tuple[int, int]:
        return self.width, self.height # notice this order!

class SubtitleType(enum.IntEnum):
    DIALOG = 0

    @staticmethod
    def num() -> int:
        return len(SubtitleType.__members__)

class FramePoint:
    def __init__(self, index: int, timestamp: int, flags: typing.List[bool]):
        self.index: int = index
        self.timestamp: int = timestamp
        if len(flags) != SubtitleType.num():
            raise Exception("len(flags) != SubtitleTypes.num()")
        self.flags: typing.List[bool] = flags

    def toString(self) -> str:
        return "frame {} {}".format(self.index, formatTimestamp(self.timestamp))

    def toStringFull(self) -> str:
        return "frame {} {} {}".format(self.index, formatTimestamp(self.timestamp), self.flags)

class FPIR: # Frame Point Intermediate Representation
    def __init__(self):
        self.framePoints: typing.List[FramePoint] = []

    def accept(self, pazz: FPIRPass) -> typing.Any:
        # returns anything
        return pazz.apply(self)

    def genVirtualEnd(self) -> FramePoint:
        index: int = len(self.framePoints)
        timestamp: int = self.framePoints[-1].timestamp
        flags = [False] * SubtitleType.num()
        return FramePoint(index, timestamp, flags)

    def getFramePointsWithVirtualEnd(self) -> typing.List[FramePoint]:
        return self.framePoints + [self.genVirtualEnd()]

class FPIRPass(abc.ABC):
    def apply(self, fpir: FPIR):
        # returns anything
        pass

class FPIRPassRemoveNoise(FPIRPass):
    def __init__(self, type: SubtitleType, minPositiveLength: int = 10, minNegativeLength: int = 2):
        self.type: SubtitleType = type
        self.minPositiveLength: int = minPositiveLength # set to 0 to disable removing positive noises
        self.minNegativeLength: int = minNegativeLength # set to 0 to disable removing negative noises

    def apply(self, fpir: FPIR):
        for id, framePoint in enumerate(fpir.framePoints):
            minLength: int = self.minNegativeLength
            if framePoint.flags[self.type]:
                minLength = self.minPositiveLength
            l = id - minLength
            r = id + minLength
            if l < 0 or r > len(fpir.framePoints) - 1:
                continue
            length = 0
            for i in range(id - 1, l - 1, -1):
                if fpir.framePoints[i].flags[self.type] != framePoint.flags[self.type]:
                    break
                length += 1
            for i in range(id + 1, r + 1):
                if fpir.framePoints[i].flags[self.type] != framePoint.flags[self.type]:
                    break
                length += 1
            if length < minLength: # flip
                framePoint.flags[self.type] = not framePoint.flags[self.type]

class FPIRPassBuildIntervals(FPIRPass):
    def __init__(self, type: SubtitleType):
        self.type: SubtitleType = type

    def apply(self, fpir: FPIR) -> typing.List[Interval]:
        intervals: typing.List[Interval] = []
        lastBegin: int = 0
        state: bool = False
        for framePoint in fpir.getFramePointsWithVirtualEnd():
            if not state: # off -> on
                if framePoint.flags[self.type]:
                    state = True
                    lastBegin = framePoint.timestamp
            else: # on - > off
                if not framePoint.flags[self.type]:
                    state = False
                    intervals.append(Interval(lastBegin, framePoint.timestamp, self.type))
        return intervals

class Interval:
    def __init__(self, begin: int, end: int, type: SubtitleType):
        self.begin: int = begin # timestamp
        self.end: int = end # timestamp
        self.type: SubtitleType = type

    def toAss(self, tag: str = "unknown") -> str:
        template = "Dialogue: 0,{},{},Default,,0,0,0,,SUBTITLE_{}_{}"
        sBegin = formatTimestamp(self.begin)
        sEnd = formatTimestamp(self.end)
        return template.format(sBegin, sEnd, self.type.name, tag)

    def dist(self, other: Interval) -> int:
        l = self
        r = other
        if self.begin > other.begin:
            l = other
            r = self
        return r.begin - l.end

    def intersects(self, other: Interval) -> bool:
        return self.dist(other) < 0

    def touches(self, other: Interval) -> bool:
        return self.dist(other) == 0

class IIR: # Interval Intermediate Representation
    def __init__(self, fpir: FPIR):
        self.intervals: typing.List[Interval] = []
        for type in SubtitleType:
            fpirPass = FPIRPassBuildIntervals(type)
            self.intervals.extend(fpir.accept(fpirPass))
        self.sort()

    def accept(self, pazz: IIRPass):
        # returns anything
        return pazz.apply(self)

    def sort(self):
        self.intervals.sort(key=lambda interval: interval.begin)
        
    def toAss(self) -> str:
        ass = ""
        for id, interval in enumerate(self.intervals):
            ass += interval.toAss(str(id)) + "\n"
        return ass

class IIRPass(abc.ABC):
    def apply(self, iir: IIR) -> typing.Any:
        # returns anything
        return 0

class IIRPassFillGap(IIRPass):
    def __init__(self, type: SubtitleType, maxGap: int = 300):
        self.type: SubtitleType = type
        self.maxGap: int = maxGap # in millisecs
    
    def apply(self, iir: IIR):
        for id, interval in enumerate(iir.intervals):
            if interval.type != self.type:
                continue
            otherId = id + 1
            while otherId < len(iir.intervals):
                otherInterval = iir.intervals[otherId]
                if otherInterval.type != self.type:
                    otherId += 1
                    continue
                if interval.dist(otherInterval) > self.maxGap:
                    break
                if interval.dist(otherInterval) <= 0:
                    otherId += 1
                    continue
                mid = (interval.end + otherInterval.begin) // 2
                interval.end = mid
                otherInterval.begin = mid
                break
        iir.sort()

def formatTimestamp(timestamp: float) -> str:
    dTimestamp = datetime.datetime.fromtimestamp(timestamp / 1000, datetime.timezone(datetime.timedelta()))
    return dTimestamp.strftime("%H:%M:%S.%f")[:-4]

def inRange(frame: cv.Mat, lower: typing.List[int], upper: typing.List[int]):
    # just a syntactic sugar
    return cv.inRange(frame, np.array(lower), np.array(upper))

def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--src", type=str, default="src.mp4", help="source video file")
    parser.add_argument("--ass", type=str, default="template.ass", help="source ass template file")
    parser.add_argument("--topblackbar", type=float, default=0.0, help="height ratio of black bar on the top of canvas, bottom is assumed symmetric")
    parser.add_argument("--leftblackbar", type=float, default=0.0, help="width ratio of black bar on the left of canvas, right is assumed symmetric")
    parser.add_argument("--dst", type=str, default="MagiaTimelineOutput.ass", help="destination ass subtitle file")
    parser.add_argument("--debug", default=False, action="store_true", help="for debugging only, show frames with debug info and save to debug.mp4")
    parser.add_argument("--shortcircuit", default=False, action="store_true", help="accelerates the program by skipping detecting other types of subtitles once one type has been confirmed, not compatible with debug mode")
    args = parser.parse_args()
    if True: # data validity test
        srcMp4Test = open(args.src, "rb")
        srcMp4Test.close()
        if not (args.topblackbar >= 0.0 and args.topblackbar <= 1.0 and args.leftblackbar >= 0.0 and args.leftblackbar <= 1.0):
            raise Exception("Invalid black bar ratio! ")
        if args.debug and args.shortcircuit:
            raise Exception("Debug mode is not compatible with short circuit mode! ")
    
    srcMp4 = cv.VideoCapture(args.src)
    srcRect = SrcRect(srcMp4)
    contentRect = RatioRect(srcRect, args.topblackbar, 1.0 - args.topblackbar, args.leftblackbar, 1.0 - args.leftblackbar)
    fps: float = srcMp4.get(cv.CAP_PROP_FPS)
    size: typing.Tuple[int, int] = contentRect.getSize()

    debugMp4: typing.Any = None
    if args.debug:
        debugMp4 = cv.VideoWriter('debug.mp4', cv.VideoWriter_fourcc('m','p','4','v'), fps, size)
    templateAss = open(args.ass, "r")
    dstAss = open(args.dst, "w")
    dstAss.writelines(templateAss.readlines())
    templateAss.close()

    dialogRect = RatioRect(contentRect, 0.68, 0.83, 0.18, 0.82)

    print("==== FPIR Building ====")
    fpir = FPIR()
    while True: # Process each frame to build FPIR

        # Frame reading

        frameIndex: int = int(srcMp4.get(cv.CAP_PROP_POS_FRAMES))
        timestamp: int = int(srcMp4.get(cv.CAP_PROP_POS_MSEC))
        validFrame, frame = srcMp4.read()
        if not validFrame:
            break

        isValidDialog: bool = False
        hasDialogBgColour: bool = False
        hasDialogText: bool = False

        while True: # For short circuit breaking

            # Dialog detection

            roiDialog = dialogRect.cutRoi(frame)
            roiDialogGray = cv.cvtColor(roiDialog, cv.COLOR_BGR2GRAY)
            _, roiDialogTextBin = cv.threshold(roiDialogGray, 172, 255, cv.THRESH_BINARY)
            roiDialogTextBinDialate = cv.morphologyEx(roiDialogTextBin, cv.MORPH_DILATE, kernel=cv.getStructuringElement(cv.MORPH_RECT, (3, 3)))
            roiDialogNoText = cv.bitwise_and(roiDialog, roiDialog, mask=255-roiDialogTextBinDialate)
            roiDialogNoTextHSV = cv.cvtColor(roiDialogNoText, cv.COLOR_BGR2HSV)
            meanDialogNoTextHSV = cv.mean(roiDialogNoTextHSV)
            meanDialogTextBin: float = cv.mean(roiDialogTextBin)[0]
            hasDialogBgColour: bool = meanDialogNoTextHSV[0] > 10 and meanDialogNoTextHSV[0] < 45 \
                and meanDialogNoTextHSV[1] > 20 and meanDialogNoTextHSV[1] < 100 \
                and meanDialogNoTextHSV[2] > 10 and meanDialogNoTextHSV[2] < 45
            hasDialogText: bool = meanDialogTextBin > 0.5 and meanDialogTextBin < 30.0

            isValidDialog = hasDialogBgColour and hasDialogText

            break

        # Frame point building

        framePoint = FramePoint(frameIndex, timestamp, [isValidDialog])
        fpir.framePoints.append(framePoint)

        # Outputs

        if frameIndex % 1000 == 0:
            print(framePoint.toString())

        if args.debug:
            frameOut = contentRect.cutRoi(frame)
            if isValidDialog:
                frameOut = cv.putText(frameOut, "VALID DIALOG", (50, 50), cv.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)
            if hasDialogBgColour:
                frameOut = cv.putText(frameOut, "has dialog bg colour", (50, 75), cv.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
            if hasDialogText:
                frameOut = cv.putText(frameOut, "has dialog text", (50, 100), cv.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
            cv.imshow("show", frameOut)
            if cv.waitKey(1) == ord('q'):
                break
            print("debug frame", frameIndex, formatTimestamp(timestamp), meanDialogNoTextHSV)
            debugMp4.write(frameOut)
    srcMp4.release()
    if args.debug:
        debugMp4.release()

    print("==== FPIR Passes ====")
    print("fpirPassRemoveNoiseDialog")
    fpirPassRemoveNoiseDialog = FPIRPassRemoveNoise(SubtitleType.DIALOG, minNegativeLength=0)
    fpir.accept(fpirPassRemoveNoiseDialog)

    print("==== FPIR to IIR ====")
    iir = IIR(fpir)

    print("==== IIR Passes ====")
    print("iirPassFillGapDialog")
    iirPassFillGapDialog = IIRPassFillGap(SubtitleType.DIALOG, 300)
    iir.accept(iirPassFillGapDialog)

    print("==== IIR to ASS ====")
    dstAss.write(iir.toAss())
    dstAss.close()

if __name__ == "__main__":
    main()
