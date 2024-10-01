import torch
import numpy as np
import matplotlib.pyplot as plt
import cv2
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator

# Tải hình ảnh đầu vào
image_path = r"C:\Users\Admin\Downloads\satelite.jpg"
image = cv2.imread(image_path)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Tải mô hình SAM
sam_checkpoint = r"D:\Year3\deeplearning\segment-anything\model\sam_vit_h_4b8939.pth"
device = "cuda" if torch.cuda.is_available() else "cpu"
sam = sam_model_registry["vit_h"](checkpoint=sam_checkpoint)
sam.to(device)

# Sử dụng bộ tạo mặt nạ tự động để phân đoạn zero-shot
mask_generator = SamAutomaticMaskGenerator(sam)
masks = mask_generator.generate(image)

# Hiển thị các phân đoạn được tạo
def show_masks(image, masks):
    if len(masks) == 0:
        print("Không tìm thấy mặt nạ.")
        return
    
    plt.figure(figsize=(10, 10))
    plt.imshow(image)
    
    for mask in masks:
        color = np.random.random((1, 3)).tolist()[0]  # Màu ngẫu nhiên cho mỗi mặt nạ
        show_mask(mask['segmentation'], color)
    
    plt.axis('off')
    plt.show()

def show_mask(mask, color):
    mask_image = np.zeros((mask.shape[0], mask.shape[1], 3))
    for i in range(3):
        mask_image[:, :, i] = mask * color[i]
    plt.imshow(np.dstack((mask_image, mask * 0.35)))

# Hiển thị kết quả
show_masks(image, masks)
