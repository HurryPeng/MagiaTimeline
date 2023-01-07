from __future__ import annotations
import abc
import typing
import numpy as np
import cv2 as cv
import datetime

class AbstractRectangle(abc.ABC):
    def getNest(self) -> typing.Tuple[int, int, float, float, float, float]:
        # returns: height, width, topOffset, leftOffset, verticalCompressRate, horizontalCompressRate
        pass

    def cutRoi(self, frame: cv.Mat) -> cv.Mat:
        pass

class RatioRectangle(AbstractRectangle):
    def __init__(self, parent: AbstractRectangle, topRatio: float, bottomRatio: float, leftRatio: float, rightRatio: float) -> None:
        self.parent: AbstractRectangle = parent
        self.topRatio: float = topRatio
        self.bottomRatio: float = bottomRatio
        self.leftRatio: float = leftRatio
        self.rightRatio: float = rightRatio

        # Precalculate offsets
        height, width, topOffset, leftOffset, verticalCompressRate, horizontalCompressRate = self.parent.getNest()
        self.height: int = height
        self.width: int = width
        self.topOffset: float = topOffset + self.height * verticalCompressRate * self.topRatio
        self.bottomOffset: float = topOffset + self.height * verticalCompressRate * self.bottomRatio
        self.leftOffset: float = leftOffset + self.width * horizontalCompressRate * self.leftRatio
        self.rightOffset: float = leftOffset + self.width * horizontalCompressRate * self.rightRatio
        self.verticalCompressRate: float = verticalCompressRate * (self.bottomRatio - self.topRatio)
        self.horizontalCompressRate: float = horizontalCompressRate * (self.rightRatio - self.leftRatio)

    def getNest(self) -> typing.Tuple[int, int, float, float, float, float]:
        return self.height, self.width, self.topOffset, self.leftOffset, self.verticalCompressRate, self.horizontalCompressRate

    def cutRoi(self, frame: cv.Mat) -> cv.Mat:
        return frame[int(self.topOffset):int(self.bottomOffset), int(self.leftOffset):int(self.rightOffset)]

class FullscreenRectangle(AbstractRectangle):
    def __init__(self, src: cv.VideoCapture):
        self.height: int = int(src.get(cv.CAP_PROP_FRAME_HEIGHT))
        self.width: int = int(src.get(cv.CAP_PROP_FRAME_WIDTH))

    def getNest(self) -> typing.Tuple[int, int, float, float, float, float]:
        return self.height, self.width, 0.0, 0.0, 1.0, 1.0

    def cutRoi(self, frame: cv.Mat) -> cv.Mat:
        return frame

def dialogBgHSVThreshold(frame: cv.Mat) -> cv.Mat:
    lower = np.array([0,   0,  160])
    upper = np.array([255, 32, 255])
    return cv.inRange(frame, lower, upper)

def dialogOutlineHSVThreshold(frame: cv.Mat) -> cv.Mat:
    lower = np.array([10,  40, 90 ])
    upper = np.array([30, 130, 190])
    return cv.inRange(frame, lower, upper)

def formatTimestamp(timestamp: float) -> str:
    dTimestamp = datetime.datetime.fromtimestamp(timestamp / 1000, datetime.timezone(datetime.timedelta()))
    return dTimestamp.strftime("%H:%M:%S.%f")[:-4]

def formatTimeline(begin: float, end: float, count: int) -> str:
    template = "Dialogue: 0,{},{},Default,,0,0,0,,VALID_SUBTITLE_{}\n"
    sBegin = formatTimestamp(begin)
    sEnd = formatTimestamp(end)
    return template.format(sBegin, sEnd, count)

def main():
    src = cv.VideoCapture("JanShizuka.mp4")

    width = int(src.get(cv.CAP_PROP_FRAME_WIDTH))
    height = int(src.get(cv.CAP_PROP_FRAME_HEIGHT))
    size = (width, height)

    fullscreenRectangle = FullscreenRectangle(src)
    contentRectangle = RatioRectangle(fullscreenRectangle, 0.09, 0.91, 0.0, 1.0) # cuts black boarders
    dialogOutlineRectangle = RatioRectangle(contentRectangle, 0.60, 0.95, 0.25, 0.75)
    dialogBgRectangle = RatioRectangle(contentRectangle, 0.7264, 0.8784, 0.3125, 0.6797)
    blackscreenRectangle = RatioRectangle(contentRectangle, 0.00, 1.00, 0.15, 0.85)

    # dst = cv.VideoWriter('out.mp4', cv.VideoWriter_fourcc('m','p','4','v'), 29, size)
    templateFile = open("template.ass", "r")
    timelineFile = open("MagiaTimelineOutput.ass", "w")
    timelineFile.writelines(templateFile.readlines())
    templateFile.close()
    
    frameCount = 0
    subtitleCount = 0
    saturatedCounter = 0 # -1: switch to invalid subtitle, 1: switch to valid subtitle
    validSubtitleState = False
    lastBeginTimestamp = 0.0

    while True:

        # Frame reading

        valid, frame = src.read()
        frameCount += 1
        if not valid:
            break
        timestamp: float = src.get(cv.CAP_PROP_POS_MSEC)
        
        # ROI selection and transformation

        roiDialogBg = dialogBgRectangle.cutRoi(frame)
        roiDialogBgHSV = cv.cvtColor(roiDialogBg, cv.COLOR_BGR2HSV)
        roiDialogBgGray = cv.cvtColor(roiDialogBg, cv.COLOR_BGR2GRAY)

        roiDialogOutline = dialogOutlineRectangle.cutRoi(frame)
        roiDialogOutlineHSV = cv.cvtColor(roiDialogOutline, cv.COLOR_BGR2HSV)

        roiBlackscreen = blackscreenRectangle.cutRoi(frame)
        roiBlackscreenGray = cv.cvtColor(roiBlackscreen, cv.COLOR_BGR2GRAY)

        # Thresholding

        _, roiDialogBgTextBin = cv.threshold(roiDialogBgGray, 192, 255, cv.THRESH_BINARY)
        roiDialogBgBin = dialogBgHSVThreshold(roiDialogBgHSV)
        roiDialogOutlineBin = dialogOutlineHSVThreshold(roiDialogOutlineHSV)
        _, roiBlackscreenBin = cv.threshold(roiBlackscreenGray, 80, 255, cv.THRESH_BINARY)
        _, roiBlackscreenTextBin = cv.threshold(roiBlackscreenGray, 160, 255, cv.THRESH_BINARY)

        # Scalarize

        meanDialogTextBin: float = cv.mean(roiDialogBgTextBin)[0]
        meanDialogBgBin: float = cv.mean(roiDialogBgBin)[0]
        meanDialogOutlineBin: float = cv.mean(roiDialogOutlineBin)[0]
        meanBlackscreenBin: float = cv.mean(roiBlackscreenBin)[0]
        meanBlackscreenTextBin: float = cv.mean(roiBlackscreenTextBin)[0]

        # Binarize

        hasDialogBg = meanDialogBgBin > 160
        hasDialogText = meanDialogTextBin < 254 and meanDialogTextBin > 200
        hasDialogOutline = meanDialogOutlineBin > 3
        isValidDialog = hasDialogBg and hasDialogText and hasDialogOutline

        hasBlackscreenBg = meanBlackscreenBin < 10
        hasBlackscreenText = meanBlackscreenTextBin > 0.1 and meanBlackscreenTextBin < 16
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
                timelineFile.write(formatTimeline(lastBeginTimestamp, timestamp, subtitleCount))
                subtitleCount += 1

        if frameCount % 1000 == 0:
            print("frame", frameCount, formatTimestamp(timestamp))

        # frameOut = frame
        # if isValidDialog:
        #     frameOut = cv.putText(frameOut, "VALID DIALOG", (50, 50), cv.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)
        # if hasDialogBg:
        #     frameOut = cv.putText(frameOut, "has dialog bg", (50, 75), cv.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
        # if hasDialogOutline:
        #     frameOut = cv.putText(frameOut, "has dialog outline", (50, 100), cv.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
        # if hasDialogText:
        #     frameOut = cv.putText(frameOut, "has dialog text", (50, 125), cv.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
        # if isValidBlackscreen:
        #     frameOut = cv.putText(frameOut, "VALID BLACKSCREEN", (50, 175), cv.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)
        # if hasBlackscreenBg:
        #     frameOut = cv.putText(frameOut, "has blackscreen bg", (50, 200), cv.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
        # if hasBlackscreenText:
        #     frameOut = cv.putText(frameOut, "has blackscreen text", (50, 225), cv.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
        # cv.imshow("show", roiDialogBgTextBin)
        # if cv.waitKey(1) == ord('q'):
        #     break
        # dst.write(frameOut)
    
    src.release()
    # dst.release()
    timelineFile.close()

if __name__ == "__main__":
    main()
