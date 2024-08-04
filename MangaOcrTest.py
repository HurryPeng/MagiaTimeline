import PIL.Image
import time

from manga_ocr import MangaOcr

mocr = MangaOcr()
img = PIL.Image.open('./test6.png')

print('Start OCR Test')

start = time.time()

text = mocr(img)

# timing end

end = time.time()

print(text)
print('Time: ', end - start)

