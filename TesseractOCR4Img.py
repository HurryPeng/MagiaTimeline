import cv2
import pytesseract
from pytesseract import Output

def process_image(img):
    """
    Processes an image to detect and draw bounding boxes around text using Tesseract.

    Args:
        img: The input image as a NumPy array.

    Returns:
        The image with bounding boxes overlayed.
    """

    # Convert the image to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Perform OCR using Tesseract
    custom_config = r' -l eng --psm 6'  # Use LSTM engine and assume a single uniform block of text
    d = pytesseract.image_to_data(gray, config=custom_config, output_type=Output.DICT)

    # Iterate over detected text boxes and draw rectangles
    n_boxes = len(d['level'])
    for i in range(n_boxes):
        if int(d['conf'][i]) > 50:  # Confidence threshold
            if (d['text'] == ""):  # Clean out empty characters
                continue
            (x, y, w, h) = (d['left'][i], d['top'][i], d['width'][i], d['height'][i])
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)  # Green bounding boxes

    return img

# Example usage
img = cv2.imread("Test11.png")
img_with_boxes = process_image(img)

# Create a larger window
cv2.namedWindow("Image with Text Detections", cv2.WINDOW_NORMAL)

# Display the image
cv2.imshow("Image with Text Detections", img)


cv2.waitKey(0)
cv2.destroyAllWindows()