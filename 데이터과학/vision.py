# # 
# import matplotlib.pyplot as plt
# import cv2

# sample_image = cv2.imread('image.jpeg')

# # opencv는 BGR로 구성되어 있음.
# # [행, 열, 채널] 순서
# blue_channel = sample_image[:, :, 0]
# green_channel = sample_image[:, :, 1]
# red_channel = sample_image[:, :, 2]

# plt.figure(figsize=(15, 3))
# plt.rc('font', family='Apple SD Gothic Neo')

# plt.subplot(1, 4, 1)
# plt.imshow(cv2.cvtColor(sample_image, cv2.COLOR_BGR2RGB))
# plt.title('원본 이미지')
# plt.axis('off')

# plt.subplot(1, 4, 2)
# plt.imshow(red_channel, cmap='Reds')
# plt.title('Red 채널')
# plt.axis('off')

# plt.subplot(1, 4, 3)
# plt.imshow(green_channel, cmap='Greens')
# plt.title('Green 채널')
# plt.axis('off')

# plt.subplot(1, 4, 4)
# plt.imshow(blue_channel, cmap='Blues')
# plt.title('Blue 채널')
# plt.axis('off')




# 이미지 생성 ====================================
import numpy as np
import matplotlib.pyplot as plt
import cv2
import time

plt.rc('font', family='Apple SD Gothic Neo')

def create_sample_image():
    height, width = 300, 400
    image = np.zeros((height, width, 3), dtype=np.uint8)

    for i in range(height):
        for j in range(width):
            image[i, j] = [i*255 // height, j*255 // width, (i+j)*255 // (height+width)]
    
    
    return image

# sample = create_sample_image()
# plt.figure(figsize=(8, 6))
# plt.imshow(cv2.cvtColor(sample, cv2.COLOR_BGR2RGB))
# plt.title('OpenCV로 생성한 샘플 이미지')
# plt.axis('off')
# plt.show()


# # 이미지 회전 --------------
# def rotate_image_examples():
#     original = create_sample_image()
#     height, width = original.shape[:2]
#     center = (width // 2, height // 2)

#     angles = [0, 45, 90, 135, 180, 270]

#     plt.figure(figsize=(18, 6))

#     for i, angle in enumerate(angles):
#         rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
#         rotated = cv2.warpAffine(original, rotation_matrix, (width, height))

#         plt.subplot(2, 3, i+1)
#         plt.imshow(cv2.cvtColor(rotated, cv2.COLOR_BGR2RGB))
#         plt.title(f'{angle}도 회전')
#         plt.axis('off')

#     plt.tight_layout()
#     plt.show()

# rotate_image_examples()


# def rotate_without_crop(image, angle):
#     height, width = image.shape[:2]
#     center = (width // 2, height // 2)

#     rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
#     cos_val = np.abs(rotation_matrix[0, 0])
#     sin_val = np.abs(rotation_matrix[0, 1])

#     new_width = int((height * sin_val) + (width * cos_val))
#     new_height = int((height * cos_val) + (width * sin_val))

#     rotation_matrix[0, 2] += (new_width / 2) - center[0]
#     rotation_matrix[1, 2] += (new_height / 2) - center[1]

#     rotated = cv2.warpAffine(image, rotation_matrix, (new_width, new_height))
#     return rotated

# original = create_sample_image()
# angle = 45

# normal_rotated = cv2.warpAffine(
#     original,
#     cv2.getRotationMatrix2D((original.shape[1]//2, original.shape[0]//2), angle, 1.0),
#     (original.shape[1], original.shape[0])
# )

# no_crop_rotated = rotate_without_crop(original, angle)

# plt.figure(figsize=(15, 5))
# plt.subplot(1, 3, 1)
# plt.imshow(cv2.cvtColor(original, cv2.COLOR_BGR2RGB))
# plt.title(f'원본')
# plt.axis('off')

# plt.subplot(1, 3, 1)
# plt.imshow(cv2.cvtColor(original, cv2.COLOR_BGR2RGB))
# plt.title(f'원본')
# plt.axis('off')

# plt.subplot(1, 3, 2)
# plt.imshow(cv2.cvtColor(normal_rotated, cv2.COLOR_BGR2RGB))
# plt.title(f'일반 회전 (잘림)')
# plt.axis('off')

# plt.subplot(1, 3, 3)
# plt.imshow(cv2.cvtColor(no_crop_rotated, cv2.COLOR_BGR2RGB))
# plt.title(f'잘림 없는 회전')
# plt.axis('off')

# plt.tight_layout()
# plt.show()

# rotate_image_examples()



# 이미지 필터링 ---------------------------------------
def demonstrate_filters():
    original = create_sample_image()
    noisy = original.copy()

    noise = np.random.normal(0, 25, original.shape).astype(np.uint8)
    noisy = cv2.add(original, noise)

    filters = {
        '원본': original,
        '노이즈 추가': noisy,
        '가우시안 블러': cv2.GaussianBlur(noisy, (15, 15), 0),
        '미디언 필터': cv2.medianBlur(noisy, 5),
        '바이래터럴 필터': cv2.bilateralFilter(noisy, 9, 75, 75),
        '샤프닝': None
    }

    sharpening_kernel = np.array([[-1, -1, -1],
                                  [-1, 9, -1], 
                                  [-1, -1, -1]])
    filters['샤프닝'] = cv2.filter2D(original, -1, sharpening_kernel)

    plt.figure(figsize=(18, 12))

    for i, (name, filtered_img) in enumerate(filters.items()):
        plt.subplot(2, 3, i+1)
        plt.imshow(cv2.cvtColor(filtered_img, cv2.COLOR_BGR2RGB))
        plt.title(name)
        plt.axis('off')

    plt.tight_layout()
    plt.show()


