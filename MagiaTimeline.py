import numpy as np
import cv2 as cv

dialogBoxUpRatio    = 0.7264
dialogBoxDownRatio  = 0.8784
dialogBoxLeftRatio  = 0.3125
dialogBoxRightRatio = 0.6797

dialogOutlineUpRatio    = 0.60
dialogOutlineDownRatio  = 0.95
dialogOutlineLeftRatio  = 0.25
dialogOutlineRightRatio = 0.75

def dialogBgHSVThreshold(frame: cv.Mat) -> cv.Mat:
    lower = np.array([0,   0,  160])
    upper = np.array([255, 32, 255])
    return cv.inRange(frame, lower, upper)

def dialogOutlineHSVThreshold(frame: cv.Mat) -> cv.Mat:
    lower = np.array([10,  40, 90 ])
    upper = np.array([30, 130, 190])
    return cv.inRange(frame, lower, upper)

def main():
    src = cv.VideoCapture("sample2-jj.mp4")

    width = int(src.get(cv.CAP_PROP_FRAME_WIDTH))
    height = int(src.get(cv.CAP_PROP_FRAME_HEIGHT))
    size = (width, height)

    dialogBoxUp    = int(height * dialogBoxUpRatio)
    dialogBoxDown  = int(height * dialogBoxDownRatio)
    dialogBoxLeft  = int(width  * dialogBoxLeftRatio)
    dialogBoxRight = int(width  * dialogBoxRightRatio)

    dialogOutlineUp    = int(height * dialogOutlineUpRatio)
    dialogOutlineDown  = int(height * dialogOutlineDownRatio)
    dialogOutlineLeft  = int(width  * dialogOutlineLeftRatio)
    dialogOutlineRight = int(width  * dialogOutlineRightRatio)

    dst = cv.VideoWriter('out.mp4', cv.VideoWriter_fourcc('m','p','4','v'), 29, size)
    
    frameCount = 0
    while True:

        valid, frame = src.read()
        frameCount += 1
        if not valid:
            break
        
        roiBox = frame[dialogBoxUp : dialogBoxDown, dialogBoxLeft : dialogBoxRight]
        roiBoxHSV = cv.cvtColor(roiBox, cv.COLOR_BGR2HSV)
        roiBoxGray = cv.cvtColor(roiBox, cv.COLOR_BGR2GRAY)

        roiOutline = frame[dialogOutlineUp : dialogOutlineDown, dialogOutlineLeft : dialogOutlineRight]
        roiOutlineHSV = cv.cvtColor(roiOutline, cv.COLOR_BGR2HSV)

        _, roiBoxTextBin = cv.threshold(roiBoxGray, 192, 255, cv.THRESH_BINARY)
        roiBoxDialogBgBin = dialogBgHSVThreshold(roiBoxHSV)
        roiBoxDialogOutlineBin = dialogOutlineHSVThreshold(roiOutlineHSV)

        meanGray: float = cv.mean(roiBoxGray)[0]
        meanDialogTextBin: float = cv.mean(roiBoxTextBin)[0]
        meanDialogBgBin: float = cv.mean(roiBoxDialogBgBin)[0]
        meanDialogOutlineBin: float = cv.mean(roiBoxDialogOutlineBin)[0]

        hasDialogBg = meanDialogBgBin > 160
        hasDialogText = meanDialogTextBin < 254 and meanDialogTextBin > 200
        hasDialogOutline = meanDialogOutlineBin > 3
        isValidDialog = hasDialogBg and hasDialogText and hasDialogOutline

        print(frameCount, meanDialogBgBin, meanDialogTextBin, meanDialogOutlineBin)

        frameOut = frame
        if isValidDialog:
            frameOut = cv.putText(frameOut, "VALID DIALOG", (50, 50), cv.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)
        if hasDialogBg:
            frameOut = cv.putText(frameOut, "has dialog bg", (50, 75), cv.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
        if hasDialogOutline:
            frameOut = cv.putText(frameOut, "has dialog outline", (50, 100), cv.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
        if hasDialogText:
            frameOut = cv.putText(frameOut, "has dialog text", (50, 125), cv.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
        cv.imshow("show", frameOut)
        if cv.waitKey(1) == ord('q'):
            break
        dst.write(frameOut)
    
    src.release()
    dst.release()

if __name__ == "__main__":
    main()