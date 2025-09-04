# # OCR ===================================


# import numpy as np
# import cv2
# import pytesseract


# def create_sample_image():

    
#     image = np.ones((200,600,3),dtype=np.uint8) * 255


#     font = cv2.FONT_HERSHEY_SIMPLEX

#     cv2.putText(image,'Test OCR',(50,100),font,2,(0,0,0),3)
#     cv2.putText(image,'Preprocessing',(50,150),font,1,(0,0,0),2)
#     cv2.imwrite('sample_text.jpg', image)

#     return image

# def basic_ocr_example():
#     image_path = "sample_text.jpg"
#     image = cv2.imread(image_path)

#     if image is None:
#         print("이미지를 찾을 수 없음.")
#         image = create_sample_image()

#     text = pytesseract.image_to_string(image,lang='eng')

#     print("인식된 텍스트:")
#     print(text)

#     return text,image

# text, image = basic_ocr_example()

# def create_high_quality_sample():
#     image = np. ones((100,400,3),dtype=np.uint8) * 255
#     font = cv2.FONT_HERSHEY_SIMPLEX
#     cv2.putText(image,'image1',(20,60),font,1.2,(0,0,0),2)
#     cv2.imwrite('high_quality_sample.jpg',image)
#     return image

# def create_medium_quality_sample():
#     image = np.ones((100,400,3),dtype=np.uint8) *255

#     noise = np.random.normal(0,15,image.shape).astype(np.uint8)
#     image =cv2.add(image, noise)

#     font = cv2.FONT_HERSHEY_SIMPLEX
#     cv2.putText(image,'Noisy OCR Text',(20,60),font,1.2,(0,0,0),2)
#     cv2.imwrite('medium_quality_sample.jpg',image)
#     return image

# def create_low_quality_sample():
#     image = np.ones((100,400,3),dtype=np.uint8) * 255

#     noise = np.random.normal(0,30,image.shape).astype(np.uint8)
#     image = cv2.add(image, noise)
#     font = cv2.FONT_HERSHEY_SIMPLEX
    
#     cv2.putText(image,'Blurry OCR Text',(20,60),font, 1.2,(50,50,50),1)
#     rows, cols = image.shape[:2] 

#     M = cv2.getRotationMatrix2D((cols/2,rows/2),5,1)
#     image = cv2.warpAffine(image,M,(cols, rows))

#     image = cv2.GaussianBlur(image,(3,3),0)
#     cv2.imwrite('low_quality_sample.jpg',image)
#     return image



# # OCR 2 ============================
# import pytesseract
# import cv2
# import matplotlib.pyplot as plt
# import numpy as np



# class OCRPreprocessor:

#     def __init__(self):
#         pass


#     def convert_to_grayscale(self,image):
#         if len(image.shape) == 3:
#             gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
#         else:
#             gray = image.copy()
#         return gray
    
#     def apply_threshold(self, image, method = 'adaptive'):
#         gray = self.convert_to_grayscale(image)
#         if method == 'simple':
#             _,thresh = cv2.threshold(gray,127,255,cv2.THRESH_BINARY)
#         elif method == 'adaptive':
#             thresh = cv2.adaptiveThreshold(
#                 gray,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
#                 cv2.THRESH_BINARY,11,2
#             )
#         elif method == 'otsu':
#             _,thresh = cv2.threshold(
#                 gray,0,255,cv2.THRESH_BINARY +cv2.THRESH_OTSU
#             )
#         return thresh

#     def resize_image(self,image,target_height = 800):
#         h, w = image.shape[:2]
#         if h < target_height:
#             scale = target_height/h
#             new_w = int(w*scale)

#             resized = cv2.resize(image,(new_w,target_height),
#                                  interpolation=cv2.INTER_CUBIC)
#         else:
#             resized = image

#         return resized
    
#     def visualize_preprocessing_steps(self, steps, step_names):

#         plt.rc('font', family = 'Apple SD Gothic Neo')
#         fig, axes = plt.subplots(2,3,figsize=(15,10))

#         axes = axes.ravel()

#         for i,(step,name) in enumerate(zip(steps,step_names)):
#             if i< len(axes):
#                 if len(step.shape) == 3:
#                     axes[i].imshow(cv2.cvtColor(step,cv2.COLOR_BGR2RGB))
#                 else:
#                     axes[i].imshow(step,cmap = 'gray')
#                 axes[i].set_title(name)
#                 axes[i].axis('off')
#         plt.tight_layout()
#         plt.show()

#     def remove_noise(self,image):
#         kernel = np.ones((3,3),np.uint8)

#         opening = cv2.morphologyEx(image,cv2.MORPH_OPEN,kernel)
#         closing = cv2.morphologyEx(opening,cv2.MORPH_CLOSE,kernel)
#         denoised = cv2.GaussianBlur(closing,(3,3),0)

#         return denoised

#     def correct_skew(self, image):
#         edges = cv2.Canny(image,50,150,apertureSize=3)
#         lines = cv2.HoughLines(edges,1,np.pi/180,threshold=100)

#         if lines is not None:
#             angles = []
#             for rho,theta in lines[:,0]:
#                 angle = np.degrees(theta) - 90
#                 angles.append(angle)
#                 median_angle = np.median(angles)
#                 (h,w) = image.shape[:2]
#                 center = (w//2,h//2)
#                 M = cv2.getRotationMatrix2D(center, median_angle, 1.0)
#                 rotated = cv2.warpAffine(image,M,(w,h),
#                                         flags = cv2.INTER_CUBIC,
#                                         borderMode= cv2.BORDER_REPLICATE)
#                 return rotated
#             return image


#     def preprocess_pipeline(self, image,visualize=False):
#         steps = []
#         step_names = []

#         steps.append(image.copy())
#         step_names.append('원본 이미지')

#         gray = self.convert_to_grayscale(image)
#         steps.append(gray)
#         step_names.append('그레이스케일')

#         resized = self.resize_image(gray)
#         steps.append(resized)
#         step_names.append('크기 조정')

#         thresh = self.apply_threshold(resized,method='adaptive')
#         steps.append(thresh)
#         step_names.append('이진화')

#         denoised = self.remove_noise(thresh)
#         steps.append(denoised)
#         step_names.append('노이즈 제거')

#         corrected = self.correct_skew(denoised)
#         steps.append(corrected)
#         step_names.append('기울기 보정')

#         if visualize:
#             self.visualize_preprocessing_steps(steps, step_names)
#         return corrected

# def create_noisy_sample_image():
#     image = np.ones((300,800,3), dtype=np.uint8) *255

#     font = cv2.FONT_HERSHEY_SIMPLEX

#     cv2.putText(image,'Noisy OCR Test Image', (50,100),font,1.5,(0,0,0),2)
#     cv2.putText(image,'Preprocessing improves accuracy',(50,150),font,1,(0,0,0),2)
#     cv2.putText(image,'Machine Learning & AI',(50,200), font,1,(0,0,0),2)

#     noise = np.random.randint(0,50,image.shape,dtype=np.uint8)
#     noisy_image = cv2.add(image,noise)

#     h,w = noisy_image.shape[:2]

#     center= (w//2,h//2)

#     M = cv2.getRotationMatrix2D(center,5,1.0)
#     skewed_image = cv2.warpAffine(noisy_image,M,(w,h))
#     return skewed_image

# def preprocessing_example():
#     preprocessor = OCRPreprocessor()
#     image = create_noisy_sample_image()
#     processed_image = preprocessor.preprocess_pipeline(image,visualize = True)

#     try:
#         import pytesseract
#         original_text = pytesseract.image_to_string(image)
#         processed_text = pytesseract.image_to_string(processed_image)

#         print("전처리 전 OCR 결과:")
#         print(repr(original_text))
#         print("\n 전처리 후 OCR 결과:")
#         print(repr(processed_text))

#     except ImportError:
#         print("pytesseract가 설치되지 않아 OCR 비교를 건너뜁니다.")
#         print("설치: pip install pytesseract")

#     return processed_image

# preprocessing_example()


# # Tesseract OCR ========================================
# import pytesseract
# import cv2
# import numpy as np
# from PIL import Image
# import matplotlib.pyplot as plt
# import pandas as pd

# # class TesseractOCR:

# class DocumentOCRDemo:
#     def __init__(self):
#         self.ocr = TesseractOCR()

#     def process_receipt_image(self, image_path):
#         image =cv2.imread(image_path)
#         image = self.create_sample_receipt()

#         processed = self.preprocess_receipt(image)
#         config = r'--oem 3 --psm 4 -c tessedit_char_whitelist=0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz.,:-$'
#         text = pytesseract.image_to_string(processed,config=config)

#         parsed_info = self.parse_receipt_info(text)
#         return text, parsed_info,processed
    

# CRNN -==================================================
# import torch
# import numpy as np

# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# class SynthecticTextDataset(Dataset):
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont
import string
import random

class CRNN(nn.Module):
    def __init__(self, img_height,img_width, num_chars, num_classes, rnn_hidden = 256):
        super().__init__

        self.img_height = img_height
        self.img_width = img_width
        self.num_chars = num_chars
        self.num_classes = num_classes

        self.cnn = nn.Sequential(
            nn.Conv2d(1,64,kernel_size=3,stride=1,padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace = True),
            nn.MaxPool2d(2,2),
        )
    

    