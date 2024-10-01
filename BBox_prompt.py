import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt
from segment_anything import SamPredictor, sam_model_registry, SamAutomaticMaskGenerator
from groundingdino.util.inference import load_model, load_image, predict

# Đọc ảnh từ tệp
image = cv2.imread(r"C:\Users\Admin\Downloads\satelite.jpg")
if image is None:
    raise ValueError("Error: Image could not be loaded. Please check the path.")

image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Bước 2: Tải mô hình SAM
sam_checkpoint = r"D:\Year3\deeplearning\segment-anything\model\sam_vit_h_4b8939.pth"
model_type = "vit_h"
sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
mask_generator = SamAutomaticMaskGenerator(sam)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
sam.to(device)

predictor = SamPredictor(sam)
# Tạo bounding boxes (ví dụ)
bounding_boxes = [
    #[802,351,925,516],  # Bounding box ví dụ 1
    [800,544,979,678]  # Bounding box ví dụ 2
]
# Iterate over bounding boxes và dự đoán
for bbox in bounding_boxes:
    predictor.set_image(image_rgb)
    input_box = np.array([bbox])  # Đảm bảo đúng kích thước (1, 4)
    masks, _, _ = predictor.predict(box=input_box)
    segmented_mask_box = masks[0]
    
   # Hiển thị kết quả
    plt.figure()
    plt.subplot(1, 2, 1)
    plt.imshow(image_rgb)
    plt.title("Original Image")
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.imshow(image_rgb)
    plt.imshow(segmented_mask_box, alpha=0.5, cmap='jet')
    plt.title(f"Bounding Box segmentation: {bbox}")
    plt.axis('off')
    plt.show()
    # Lưu kết quả segmentation
    segmented_mask = (segmented_mask_box * 255).astype(np.uint8)
    contours, _ = cv2.findContours(segmented_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contour_image = image_rgb.copy()
    cv2.drawContours(contour_image, contours, -1, (255, 0, 0), 2)
    cv2.imwrite("segmented_building.png", cv2.cvtColor(contour_image, cv2.COLOR_RGB2BGR))

    # Hiển thị ảnh contours
    plt.imshow(contour_image)
    plt.title("Segmented Image with Contours")
    plt.axis('off')
    plt.show()

