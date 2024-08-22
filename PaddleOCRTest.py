import cv2
from paddleocr import PaddleOCR, draw_ocr
import numpy as np
import time

# 初始化PaddleOCR，指定使用日文模型
ocr = PaddleOCR(use_angle_cls=True, lang='japan', show_log=False) 

def recognize_text(image_path):
    # 读取图像
    img = cv2.imread(image_path)
    
    # 进行OCR识别
    result: str = ocr.ocr(img, cls=True, bin=True)
    
    # 处理并输出结果
    recognizedText = ''
    for line in result:
        lineText = ''.join([wordInfo[1][0] for wordInfo in line])
        recognizedText += lineText + '\n'
    recognizedText = recognizedText.strip()

    return recognizedText

# 示例用法
image_path = 'test4.png'  # 替换为你的图像路径
recognized_text = recognize_text(image_path)
print(recognized_text)

# import cv2
# import numpy as np
# from paddleocr import PaddleOCR

# # 初始化PaddleOCR，指定使用日文模型
# ocr = PaddleOCR(use_angle_cls=True, lang='japan')

# def detect_text_area(image_path):
#     # 读取图像
#     img = cv2.imread(image_path)
    
#     # 进行文本检测（不进行识别）
#     result = ocr.ocr(img, det=True, rec=False)
    
#     # 创建一个与输入图像大小相同的黑色背景图像
#     text_mask = np.zeros_like(img, dtype=np.uint8)
    
#     # 绘制检测到的文本区域
#     for line in result:
#         points = np.array(line[0], dtype=np.int32)
#         cv2.fillPoly(text_mask, [points], (255, 255, 255))
    
#     # 将图像转换为灰度图像
#     gray_text_mask = cv2.cvtColor(text_mask, cv2.COLOR_BGR2GRAY)
    
#     return gray_text_mask

# # 示例用法
# image_path = 'test.png'  # 替换为你的图像路径

# # 运行60次以进行测速

# begin_time = time.time()

# for i in range(1, 60):
#     text_mask = detect_text_area(image_path)

# end_time = time.time()
# print('Time: ', end_time - begin_time)

# # 保存或显示结果
# cv2.imwrite('text_mask.jpg', text_mask)
# cv2.imshow('Text Area Mask', text_mask)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
