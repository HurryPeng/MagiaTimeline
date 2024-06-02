

# Page Segmentation Modes (PSM)
# 0: Orientation and script detection (OSD) only.
# 1: Automatic page segmentation with OSD.
# 2: Automatic page segmentation, but no OSD, or OCR.
# 3: Fully automatic page segmentation, but no OSD. (Default)
# 4: Assume a single column of text of variable sizes.
# 5: Assume a single uniform block of vertically aligned text.
# 6: Assume a single uniform block of text.
# 7: Treat the image as a single text line.
# 8: Treat the image as a single word.
# 9: Treat the image as a single word in a circle.
# 10: Treat the image as a single character.
# 11: Sparse text. Find as much text as possible in no particular order.
# 12: Sparse text with OSD.
# 13: Raw line. Treat the image as a single text line, bypassing hacks that are Tesseract-specific.
#


# OCR Engine Modes (OEM)
# OEM 0: Legacy engine only.
#
# This mode uses the traditional Tesseract OCR engine, which was used in older versions of Tesseract before the introduction of the neural network-based LSTM engine.
# OEM 1: Neural nets LSTM engine only.
#
# This mode uses the newer Long Short-Term Memory (LSTM) based OCR engine, which typically offers improved accuracy, especially for recognizing more complex text and fonts.
# OEM 2: Legacy + LSTM engines.
#
# This mode combines the results of both the legacy and LSTM engines, potentially improving accuracy by leveraging the strengths of both engines.
# OEM 3: Default, based on what is available.
#
# This mode allows Tesseract to automatically select the best available OCR engine. In most cases, it defaults to using the LSTM engine if it is available.


# either have to preprocess/crop the image, or train a model to identify text that is a subtitle!!!!

import cv2 as cv
import numpy as np
import pytesseract
from pytesseract import Output


# # Specify image dimensions and desired data type

# height, width = cv.imread("test2.png").shape[:2]

# print (imgs)


def process_image(img, htmps):
    # Convert the image to grayscale
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    # Optionally apply some preprocessing (e.g., thresholding) if needed
    # _, thresh = cv.threshold(gray, 150, 255, cv.THRESH_BINARY)

    # Perform OCR using pytesseract
    # tsvStr = pytesseract.image_to_data(img, lang="eng", config="--psm 11")
    # custom_config = r' -l jpn --psm 11'  # Use LSTM engine and assume a single uniform block of text
    custom_config = r' -l eng --psm 11'  # Use LSTM engine and assume a single uniform block of text
    # custom_config = r' -l jpn --oem 3 --psm 11'  # Use LSTM engine and assume a single uniform block of text
    d = pytesseract.image_to_data(gray, config=custom_config, output_type=Output.DICT)

    # print(d.items())

    # TODO: GENERATE HEATMAP OF  BOUNDING BOXES
    # Iterate over detected text boxes and draw rectangles
    n_boxes = len(d['level'])
    for i in range(n_boxes):
        if int(d['conf'][i]) > 50:  # Confidence threshold
            if (d['text'] == ""):  # clean out the empty characters
                continue
            (x, y, w, h) = (d['left'][i], d['top'][i], d['width'][i], d['height'][i])
            if x + w <= width and y + h <= height:
                # htmps[y:y+h, x:x+w] += 1
                htmps[y:y + h, x:x + w] += float(d['conf'][i])

            cv.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0),
                         2)  # top-left and bottom-right corner coordinates, (0, 255, 0) is green color, 2 is thickness
            cv.putText(img, d['text'][i], (x, y - 10), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)




# src = './202401/2024NewYearLogin.mov'
src = 'pokemon_3.mp4'

srcMp4 = cv.VideoCapture(src)
fps: float = srcMp4.get(cv.CAP_PROP_FPS)
frameCount = srcMp4.get(cv.CAP_PROP_FRAME_COUNT)

print(frameCount)

width = int(srcMp4.get(cv.CAP_PROP_FRAME_WIDTH))
height = int(srcMp4.get(cv.CAP_PROP_FRAME_HEIGHT))

dtype = np.float32  # 32-bit floating-point numbers
heatmap = np.zeros((height, width), dtype=dtype)

while True:  # Process each frame to build FPIR

    # Frame reading
    frameIndex: int = int(srcMp4.get(cv.CAP_PROP_POS_FRAMES))
    timestamp: int = int(srcMp4.get(cv.CAP_PROP_POS_MSEC))
    validFrame, frame = srcMp4.read()
    if not validFrame:
        break

    if frameIndex % 100 != 0:
        continue

    print(frameIndex)
    process_image(frame, heatmap)


np.savetxt('array_output.txt', heatmap, fmt='%d')

normalized_heatmap = cv.normalize(heatmap, None, alpha=0, beta=255, norm_type=cv.NORM_MINMAX)

cv.imwrite('poke3.png', normalized_heatmap)



# Normalize pixel values thresholding to set all pixel values to 0 and 1
# Then feed their coordinates as the feature for clustering
# doing all this for each video