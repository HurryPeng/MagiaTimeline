from __future__ import annotations
import abc
import typing
import numpy as np
import cv2 as cv
import datetime
import argparse

class AbstractRect(abc.ABC):
    def getNest(self) -> typing.Tuple[int, int, float, float, float, float]:
        # returns: fullHeight, fullWidth, topOffset, leftOffset, verticalCompressRate, horizontalCompressRate
        pass

    def cutRoi(self, frame: cv.Mat) -> cv.Mat:
        pass

    def getSize(self) -> typing.Tuple[int, int]:
        # returns: width, height
        # notice this order!
        pass

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

def formatTimestamp(timestamp: float) -> str:
    dTimestamp = datetime.datetime.fromtimestamp(timestamp / 1000, datetime.timezone(datetime.timedelta()))
    return dTimestamp.strftime("%H:%M:%S.%f")[:-4]

def formatTimeline(begin: float, end: float, count: int) -> str:
    template = "Dialogue: 0,{},{},Default,,0,0,0,,VALID_SUBTITLE_{}\n"
    sBegin = formatTimestamp(begin)
    sEnd = formatTimestamp(end)
    return template.format(sBegin, sEnd, count)

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
    args = parser.parse_args()
    if True: # data validity test
        srcMp4Test = open(args.src, "rb")
        srcMp4Test.close()
        if not (args.topblackbar >= 0.0 and args.topblackbar <= 1.0 and args.leftblackbar >= 0.0 and args.leftblackbar <= 1.0):
            raise Exception("Invalid black bar ratio! ")
    
    srcMp4 = cv.VideoCapture(args.src)
    srcRect = SrcRect(srcMp4)
    contentRect = RatioRect(srcRect, args.topblackbar, 1.0 - args.topblackbar, args.leftblackbar, 1.0 - args.leftblackbar)
    fps: float = srcMp4.get(cv.CAP_PROP_FPS)
    size: typing.Tuple[int, int] = contentRect.getSize()

    if args.debug:
        debugMp4 = cv.VideoWriter('debug.mp4', cv.VideoWriter_fourcc('m','p','4','v'), fps, size)
    templateAss = open(args.ass, "r")
    dstAss = open(args.dst, "w")
    dstAss.writelines(templateAss.readlines())
    templateAss.close()

    dialogOutlineRect = RatioRect(contentRect, 0.60, 0.95, 0.25, 0.75)
    dialogBgRect = RatioRect(contentRect, 0.7264, 0.8784, 0.3125, 0.6797)
    blackscreenRect = RatioRect(contentRect, 0.00, 1.00, 0.15, 0.85)

    frameCount = 0
    subtitleCount = 0
    saturatedCounter = 0 # -1: switch to invalid subtitle, 1: switch to valid subtitle
    validSubtitleState = False
    lastBeginTimestamp = 0.0

    while True:

        # Frame reading

        validFrame, frame = srcMp4.read()
        frameCount += 1
        if not validFrame:
            break
        timestamp: float = srcMp4.get(cv.CAP_PROP_POS_MSEC)

        # Subtitle recognizing

        roiDialogBg = dialogBgRect.cutRoi(frame)
        roiDialogBgGray = cv.cvtColor(roiDialogBg, cv.COLOR_BGR2GRAY)
        roiDialogBgHSV = cv.cvtColor(roiDialogBg, cv.COLOR_BGR2HSV)
        roiDialogBgBin = inRange(roiDialogBgHSV, [0, 0, 160], [255, 32, 255])
        _, roiDialogBgTextBin = cv.threshold(roiDialogBgGray, 192, 255, cv.THRESH_BINARY)
        meanDialogTextBin: float = cv.mean(roiDialogBgTextBin)[0]
        meanDialogBgBin: float = cv.mean(roiDialogBgBin)[0]
        hasDialogBg = meanDialogBgBin > 160
        hasDialogText = meanDialogTextBin < 254 and meanDialogTextBin > 200

        roiDialogOutline = dialogOutlineRect.cutRoi(frame)
        roiDialogOutlineHSV = cv.cvtColor(roiDialogOutline, cv.COLOR_BGR2HSV)
        roiDialogOutlineBin = inRange(roiDialogOutlineHSV, [10, 40, 90], [30, 130, 190])
        meanDialogOutlineBin: float = cv.mean(roiDialogOutlineBin)[0]
        hasDialogOutline = meanDialogOutlineBin > 3

        roiBlackscreen = blackscreenRect.cutRoi(frame)
        roiBlackscreenGray = cv.cvtColor(roiBlackscreen, cv.COLOR_BGR2GRAY)
        _, roiBlackscreenBgBin = cv.threshold(roiBlackscreenGray, 80, 255, cv.THRESH_BINARY)
        _, roiBlackscreenTextBin = cv.threshold(roiBlackscreenGray, 160, 255, cv.THRESH_BINARY)
        meanBlackscreenBgBin: float = cv.mean(roiBlackscreenBgBin)[0]
        meanBlackscreenTextBin: float = cv.mean(roiBlackscreenTextBin)[0]
        hasBlackscreenBg = meanBlackscreenBgBin < 20
        hasBlackscreenText = meanBlackscreenTextBin > 0.1 and meanBlackscreenTextBin < 16

        isValidDialog = hasDialogBg and hasDialogText and hasDialogOutline
        isValidBlackscreen = hasBlackscreenBg and hasBlackscreenText
        isValidSubtitle = isValidDialog or isValidBlackscreen

        # State machine

        if isValidSubtitle and saturatedCounter < 1:
            saturatedCounter += 1
        if not isValidSubtitle and saturatedCounter > -1:
            saturatedCounter -= 1
        
        lastValidSubtitleState = validSubtitleState
        if saturatedCounter == 1:
            validSubtitleState = True
        if saturatedCounter == -1:
            validSubtitleState = False
        
        if lastValidSubtitleState != validSubtitleState: # state changed
            if validSubtitleState == True: # flipped on
                lastBeginTimestamp = timestamp
            else: # flipped off
                dstAss.write(formatTimeline(lastBeginTimestamp, timestamp, subtitleCount))
                subtitleCount += 1

        if frameCount % 1000 == 0:
            print("frame", frameCount, formatTimestamp(timestamp))

        if args.debug:
            frameOut = contentRect.cutRoi(frame)
            if isValidDialog:
                frameOut = cv.putText(frameOut, "VALID DIALOG", (50, 50), cv.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)
            if hasDialogBg:
                frameOut = cv.putText(frameOut, "has dialog bg", (50, 75), cv.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
            if hasDialogOutline:
                frameOut = cv.putText(frameOut, "has dialog outline", (50, 100), cv.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
            if hasDialogText:
                frameOut = cv.putText(frameOut, "has dialog text", (50, 125), cv.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
            if isValidBlackscreen:
                frameOut = cv.putText(frameOut, "VALID BLACKSCREEN", (50, 175), cv.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)
            if hasBlackscreenBg:
                frameOut = cv.putText(frameOut, "has blackscreen bg", (50, 200), cv.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
            if hasBlackscreenText:
                frameOut = cv.putText(frameOut, "has blackscreen text", (50, 225), cv.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
            cv.imshow("show", frameOut)
            if cv.waitKey(1) == ord('q'):
                break
            print("debug frame", frameCount, formatTimestamp(timestamp), meanBlackscreenBgBin)
            debugMp4.write(frameOut)
    
    srcMp4.release()
    if args.debug:
        debugMp4.release()
    dstAss.close()

if __name__ == "__main__":
    main()
