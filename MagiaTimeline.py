import cv2 as cv

dialogueBoarderUp    = 0.7264
dialogueBoarderDown  = 0.8784
dialogueBoarderLeft  = 0.3125
dialogueBoarderRight = 0.6797

def main():
    src = cv.VideoCapture("sample.mp4")
    width = int(src.get(cv.CAP_PROP_FRAME_WIDTH))
    height = int(src.get(cv.CAP_PROP_FRAME_HEIGHT))
    size = (width, height)
    dst = cv.VideoWriter('out.mp4', cv.VideoWriter_fourcc('m','p','4','v'), 29, size)
    
    frameCount = 0
    while True:

        valid, frame = src.read()
        frameCount += 1
        if not valid:
            break
        
        # frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        roi = frame[int(height * dialogueBoarderUp) : int(height * dialogueBoarderDown), int(width * dialogueBoarderLeft) : int(width * dialogueBoarderRight)]
        poiGray = cv.cvtColor(roi, cv.COLOR_BGR2GRAY)
        meanGray: float = cv.mean(poiGray)[0]
        _, poiBin = cv.threshold(poiGray, 192, 255, cv.THRESH_BINARY)
        meanBin: float = cv.mean(poiBin)[0]

        isWhiteBackground = meanGray > 200
        hasText = meanBin < 254 and meanBin > 200
        isValidSubtitle = isWhiteBackground and hasText

        print(frameCount, meanGray, meanBin, isValidSubtitle)

        # framePil = Image.fromarray(cv.cvtColor(frame, cv.COLOR_BGR2RGB))
        # text = pytesseract.image_to_string(poi, "jpn")
        # print(text)
        # print(mocr(framePil))
        # print(reader.readtext(frame))
        # print(mocr(framePil))

        frameOut = frame
        if isValidSubtitle:
            frameOut = cv.putText(frameOut, "VALID SUBTITLE", (50, 50), cv.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)
        if isWhiteBackground:
            frameOut = cv.putText(frameOut, "is white bg", (50, 75), cv.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
        if hasText:
            frameOut = cv.putText(frameOut, "has text", (50, 100), cv.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
        cv.imshow("show", frameOut)
        if cv.waitKey(1) == ord('q'):
            break
        dst.write(frameOut)
    
    src.release()
    dst.release()

if __name__ == "__main__":
    main()