# import typing
import cv2 as cv
# import pytesseract
# import easyocr

# import manga_ocr
# mocr = manga_ocr.MangaOcr()
# reader = easyocr.Reader(['ja','en']) 

def main():
    src = cv.VideoCapture("sample.mp4")
    size = (1280, 592)
    dst = cv.VideoWriter('out.mp4', cv.VideoWriter_fourcc('m','p','4','v'), 29, size)
    
    frameCount = 0
    while True:

        valid, frame = src.read()
        frameCount += 1
        if not valid:
            break
        
        # frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        poi = frame[420:530, 400:870]
        poiGray = cv.cvtColor(poi, cv.COLOR_BGR2GRAY)
        meanGray: float = cv.mean(poiGray)[0]
        _, poiBin = cv.threshold(poiGray, 192, 255, cv.THRESH_BINARY)
        meanBin: float = cv.mean(poiBin)[0]

        isDialogue = meanGray > 200
        hasText = meanBin < 254
        isValidSubtitle = isDialogue and hasText

        print(frameCount, meanGray, meanBin, isValidSubtitle)

        # framePil = Image.fromarray(cv.cvtColor(frame, cv.COLOR_BGR2RGB))
        # text = pytesseract.image_to_string(poi, "jpn")
        # print(text)
        # print(mocr(framePil))
        # print(reader.readtext(frame))
        # print(mocr(framePil))

        frameOut = None
        if isValidSubtitle:
            frameOut = cv.putText(frame, "VALID SUBTITLE", (50, 50), cv.FONT_HERSHEY_SIMPLEX, 1.2, (255, 0, 0), 2)
        else:
            frameOut = frame
        cv.imshow("show", poiBin)
        if cv.waitKey(1) == ord('q'):
            break
        dst.write(frameOut)
    
    src.release()
    dst.release()

if __name__ == "__main__":
    main()