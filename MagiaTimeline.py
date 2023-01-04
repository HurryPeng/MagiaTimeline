# import typing
import cv2 as cv
from PIL import Image
import pytesseract
# import easyocr

# import manga_ocr
# mocr = manga_ocr.MangaOcr()
# reader = easyocr.Reader(['ja','en']) 

def main():
    src = cv.VideoCapture("sample.mp4")
    size = (1280, 592)
    dst = cv.VideoWriter('out.mp4', cv.VideoWriter_fourcc('m','p','4','v'), 29, size)
    
    frameCount = 0
    lastText = ""
    lastCount = 0
    while True:

        valid, frame = src.read()
        frameCount += 1
        if not valid:
            break
        
        if frameCount % 4 == 0:
            # frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
            poi = frame[420:530, 400:870]
            print(frameCount, lastCount)
            # framePil = Image.fromarray(cv.cvtColor(frame, cv.COLOR_BGR2RGB))
            text = pytesseract.image_to_string(poi, "jpn")
            print(text)
            # print(mocr(framePil))
            # print(reader.readtext(frame))
            # print(mocr(framePil))
            if lastText == text and text != "":
                lastCount += 1
            else:
                lastText = text
                lastCount = 0
        if lastCount >= 2:
            frame = cv.putText(frame, "VALID SUBTITLE", (50, 50), cv.FONT_HERSHEY_SIMPLEX, 1.2, (255, 0, 0), 2)
            print("VALID SUBTITLE")
        cv.imshow("show", frame)
        if cv.waitKey(1) == ord('q'):
            break
        dst.write(frame)
    
    src.release()
    dst.release()

if __name__ == "__main__":
    main()