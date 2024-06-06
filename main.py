# import cv2 as cv
# import pytesseract
#
# def main():
#     # Read "HomuraDetermination.png"
#     img = cv.imread("gengar.png")
#
#     # Process the image with tesseract as if running
#     # tesseract HomuraDetermination.png stdout -l jpn --psm 11
#     # output as tsv datastructure
#
#     tsvStr = pytesseract.image_to_data(img, lang="eng", config="--psm 11")
#
#     tsv_data = []
#     for line in tsvStr.splitlines():
#         if line.startswith('level'):  # Skip header line
#             continue
#
#         # Split by whitespace (adjust delimiter if needed)
#         fields = line.split()
#
#         row = {
#             'level': fields[0],
#             'page_num': int(fields[1]),
#             'block_num': int(fields[2]),
#             'par_num': int(fields[3]),
#             'line_num': int(fields[4]),
#             'word_num': int(fields[5]),
#             'left': int(fields[6]),
#             'top': int(fields[7]),
#             'width': int(fields[8]),
#             'height': int(fields[9]),
#             'conf': float(fields[10]),
#             # Some rows do not have fields[11]
#             'text': fields[11] if len(fields) == 12 else ''
#         }
#
#         tsv_data.append(row)
#
#     # delete from tsc_data where confidence is less than 30.0
#     tsv_data = [row for row in tsv_data if row['conf'] >= 40.0]
#
#     # draw red rectangles on the image where the text is
#     for row in tsv_data:
#         x, y, w, h = row['left'], row['top'], row['width'], row['height']
#         cv.rectangle(img, (x, y), (x+w, y+h), (0, 0, 255), 2)
#
#     cv.imshow("gengar", img)
#     while True:
#         if cv.waitKey(1) == ord('q'):
#             break
#
# main()

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




#either have to preprocess/crop the image, or train a model to identify text that is a subtitle!!!!

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
    custom_config = r' -l jpn --psm 11'  # Use LSTM engine and assume a single uniform block of text
    # custom_config = r' -l jpn --oem 3 --psm 11'  # Use LSTM engine and assume a single uniform block of text
    d = pytesseract.image_to_data(gray, config=custom_config, output_type=Output.DICT)

    # print(d.items())

    #TODO: GENERATE HEATMAP OF  BOUNDING BOXES
    # Iterate over detected text boxes and draw rectangles
    n_boxes = len(d['level'])
    for i in range(n_boxes):
        if int(d['conf'][i]) > 50:  # Confidence threshold
            if ( d['text'] == ""): #clean out the empty characters
                continue
            (x, y, w, h) = (d['left'][i], d['top'][i], d['width'][i], d['height'][i])
            if x + w <= width and y + h <= height:
                # htmps[y:y+h, x:x+w] += 1
                htmps[y:y+h, x:x+w] += float(d['conf'][i])

            cv.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2) # top-left and bottom-right corner coordinates, (0, 255, 0) is green color, 2 is thickness
            cv.putText(img, d['text'][i], (x, y - 10), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

src = './202312/Christmas2023.mov'
# src = './202401/2024NewYearLogin.mov'
# src = './202402/Daitoudanchi2-1.mp4'
# src = './202404/Yuu.mp4'
# src = './202405/LastBirdsHope-1.mp4'
# src = './parako/Parako-287.mp4'

srcMp4 = cv.VideoCapture(src)
fps: float = srcMp4.get(cv.CAP_PROP_FPS)
frameCount = srcMp4.get(cv.CAP_PROP_FRAME_COUNT)

print(frameCount)

width = int(srcMp4.get(cv.CAP_PROP_FRAME_WIDTH))
height = int(srcMp4.get(cv.CAP_PROP_FRAME_HEIGHT))

dtype = np.float32  # 32-bit floating-point numbers
heatmap = np.zeros((height, width), dtype=dtype)

while True: # Process each frame to build FPIR

    # Frame reading
    frameIndex: int = int(srcMp4.get(cv.CAP_PROP_POS_FRAMES))
    timestamp: int = int(srcMp4.get(cv.CAP_PROP_POS_MSEC))
    validFrame, frame = srcMp4.read()
    if not validFrame:
        break

    if frameIndex % 10 != 0:
        continue
    
    print(frameIndex)
    process_image(frame, heatmap)


    # CV and frame point building

    # np.set_printoptions(threshold=np.inf)  # Set the threshold to infinity to display the entire array
    # print(imgs)
    # np.set_printoptions(threshold=1000)
np.savetxt('array_output.txt', heatmap, fmt='%d')

normalized_heatmap = cv.normalize(heatmap, None, alpha=0, beta=255, norm_type=cv.NORM_MINMAX)

cv.imwrite('colorized_heatmap.png', normalized_heatmap)

# Display the image with highlighted subtitles
# cv.imshow('Subtitles Detection', img)
# cv.waitKey(0)
# cv.destroyAllWindows()











# if __name__ == "__main__":
#     process_image("test2.png")




#EAST TEXT DETECTOR ATTEMPT
#
# import cv2 as cv
# import numpy as np
#
#
# def detect_subtitles(image_path):
#     # Load image
#     img = cv.imread(image_path)
#     orig = img.copy()
#     (H, W) = img.shape[:2]
#
#     # Load EAST text detector
#     net = cv.dnn.readNet(r"C:\Users\Andrew Jeon\OneDrive\Desktop\MagiaTimeline\opencv-text-detection\frozen_east_text_detection.pb")
#
#     # Prepare the image for the EAST detector
#     blob = cv.dnn.blobFromImage(img, 1.0, (W, H), (123.68, 116.78, 103.94), swapRB=True, crop=False)
#     net.setInput(blob)
#
#     # Get the scores and geometry from the EAST detector
#     (scores, geometry) = net.forward(["feature_fusion/Conv_7/Sigmoid", "feature_fusion/concat_3"])
#
#     # Decode the predictions
#     rects = []
#     confidences = []
#     for i in range(scores.shape[2]):
#         for j in range(scores.shape[3]):
#             if scores[0, 0, i, j] > 0.5:
#                 offsetX, offsetY = j * 4.0, i * 4.0
#                 angle = geometry[0, 4, i, j]
#                 cos = np.cos(angle)
#                 sin = np.sin(angle)
#                 h = geometry[0, 0, i, j] + geometry[0, 2, i, j]
#                 w = geometry[0, 1, i, j] + geometry[0, 3, i, j]
#                 endX = int(offsetX + (cos * geometry[0, 1, i, j]) + (sin * geometry[0, 2, i, j]))
#                 endY = int(offsetY - (sin * geometry[0, 1, i, j]) + (cos * geometry[0, 2, i, j]))
#                 startX = int(endX - w)
#                 startY = int(endY - h)
#                 rects.append((startX, startY, endX, endY))
#                 confidences.append(scores[0, 0, i, j])
#
#     # Apply non-maxima suppression to filter overlapping boxes
#     boxes = cv.dnn.NMSBoxes(rects, confidences, 0.5, 0.4)
#
#     # Draw the bounding boxes
#     for i in boxes.flatten():
#         (startX, startY, endX, endY) = rects[i]
#         if startY > H * 0.7:  # Heuristic: consider text in the bottom 30% of the image
#             cv.rectangle(orig, (startX, startY), (endX, endY), (0, 255, 0), 2)
#
#     cv.imshow("Subtitles Detection", orig)
#     cv.waitKey(0)
#     cv.destroyAllWindows()
#
#
# if __name__ == "__main__":
#     detect_subtitles("gengar.png")