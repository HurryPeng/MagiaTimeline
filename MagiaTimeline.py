import numpy as np
import cv2 as cv

dialogBgUpRatio    = 0.7264
dialogBgDownRatio  = 0.8784
dialogBgLeftRatio  = 0.3125
dialogBgRightRatio = 0.6797

dialogOutlineUpRatio    = 0.60
dialogOutlineDownRatio  = 0.95
dialogOutlineLeftRatio  = 0.25
dialogOutlineRightRatio = 0.75

blackscreenUpRatio    = 0.1
blackscreenDownRatio  = 0.9
blackscreenLeftRatio  = 0.2
blackscreenRightRatio = 0.8

def dialogBgHSVThreshold(frame: cv.Mat) -> cv.Mat:
    lower = np.array([0,   0,  160])
    upper = np.array([255, 32, 255])
    return cv.inRange(frame, lower, upper)

def dialogOutlineHSVThreshold(frame: cv.Mat) -> cv.Mat:
    lower = np.array([10,  40, 90 ])
    upper = np.array([30, 130, 190])
    return cv.inRange(frame, lower, upper)

def main():
    src = cv.VideoCapture("sample2.mp4")

    width = int(src.get(cv.CAP_PROP_FRAME_WIDTH))
    height = int(src.get(cv.CAP_PROP_FRAME_HEIGHT))
    size = (width, height)

    dialogBgUp    = int(height * dialogBgUpRatio)
    dialogBgDown  = int(height * dialogBgDownRatio)
    dialogBgLeft  = int(width  * dialogBgLeftRatio)
    dialogBgRight = int(width  * dialogBgRightRatio)
    dialogOutlineUp    = int(height * dialogOutlineUpRatio)
    dialogOutlineDown  = int(height * dialogOutlineDownRatio)
    dialogOutlineLeft  = int(width  * dialogOutlineLeftRatio)
    dialogOutlineRight = int(width  * dialogOutlineRightRatio)
    blackscreenUp    = int(height * blackscreenUpRatio)
    blackscreenDown  = int(height * blackscreenDownRatio)
    blackscreenLeft  = int(width  * blackscreenLeftRatio)
    blackscreenRight = int(width  * blackscreenRightRatio)

    dst = cv.VideoWriter('out.mp4', cv.VideoWriter_fourcc('m','p','4','v'), 29, size)
    
    frameCount = 0
    while True:

        valid, frame = src.read()
        frameCount += 1
        if not valid:
            break
        
        roiDialogBg = frame[dialogBgUp : dialogBgDown, dialogBgLeft : dialogBgRight]
        roiDialogBgHSV = cv.cvtColor(roiDialogBg, cv.COLOR_BGR2HSV)
        roiDialogBgGray = cv.cvtColor(roiDialogBg, cv.COLOR_BGR2GRAY)

        roiDialogOutline = frame[dialogOutlineUp : dialogOutlineDown, dialogOutlineLeft : dialogOutlineRight]
        roiDialogOutlineHSV = cv.cvtColor(roiDialogOutline, cv.COLOR_BGR2HSV)

        roiBlackscreen = frame[blackscreenUp : blackscreenDown, blackscreenLeft : blackscreenRight]
        roiBlackscreenGray = cv.cvtColor(roiBlackscreen, cv.COLOR_BGR2GRAY)

        _, roiDialogBgTextBin = cv.threshold(roiDialogBgGray, 192, 255, cv.THRESH_BINARY)
        roiDialogBgBin = dialogBgHSVThreshold(roiDialogBgHSV)
        roiDialogOutlineBin = dialogOutlineHSVThreshold(roiDialogOutlineHSV)
        _, roiBlackscreenBin = cv.threshold(roiBlackscreenGray, 80, 255, cv.THRESH_BINARY)
        _, roiBlackscreenTextBin = cv.threshold(roiBlackscreenGray, 160, 255, cv.THRESH_BINARY)

        meanDialogTextBin: float = cv.mean(roiDialogBgTextBin)[0]
        meanDialogBgBin: float = cv.mean(roiDialogBgBin)[0]
        meanDialogOutlineBin: float = cv.mean(roiDialogOutlineBin)[0]
        meanBlackscreenBin: float = cv.mean(roiBlackscreenBin)[0]
        meanBlackscreenTextBin: float = cv.mean(roiBlackscreenTextBin)[0]

        hasDialogBg = meanDialogBgBin > 160
        hasDialogText = meanDialogTextBin < 254 and meanDialogTextBin > 200
        hasDialogOutline = meanDialogOutlineBin > 3
        isValidDialog = hasDialogBg and hasDialogText and hasDialogOutline

        hasBlackscreenBg = meanBlackscreenBin < 10
        hasBlackscreenText = meanBlackscreenTextBin > 0.2 and meanBlackscreenTextBin < 16
        isValidBlackscreen = hasBlackscreenBg and hasBlackscreenText

        print(frameCount, meanBlackscreenTextBin)

        frameOut = frame
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
        dst.write(frameOut)
    
    src.release()
    dst.release()

if __name__ == "__main__":
    main()