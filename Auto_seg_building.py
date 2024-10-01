import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt
from segment_anything import SamAutomaticMaskGenerator, sam_model_registry

# Load image from UC Merced dataset
image_path = r"C:\Users\Admin\Downloads\satelite.jpg"
image = cv2.imread(image_path)
if image is None:
    raise ValueError("Error: Image could not be loaded. Please check the path.")

image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Load SAM model
sam_checkpoint = r"D:\Year3\deeplearning\segment-anything\model\sam_vit_h_4b8939.pth"
model_type = "vit_h"
sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
sam.to(device)

# Generate masks automatically using SAM(zero shot)
mask_generator = SamAutomaticMaskGenerator(sam)
masks = mask_generator.generate(image_rgb)

# Initialize a mask to combine valid segments
segmented_mask = np.zeros(image_rgb.shape[:2], dtype=np.uint8)

# Define thresholds for area
threshold_area = 400  # Minimum area

# Convert image to HSV color space
image_hsv = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2HSV)

# Define lower and upper bounds for green color in HSV
lower_green = np.array([40, 40, 40])  # Hue, Saturation, Value for lower bound of green
upper_green = np.array([90, 255, 255])  # Hue, Saturation, Value for upper bound of green

# Create mask for green areas
green_mask = cv2.inRange(image_hsv, lower_green, upper_green)

# Apply criteria to filter masks
for mask in masks:
    area = mask['area']
    segmentation = mask['segmentation']

    if area > threshold_area:
        # Check if the mask intersects with the green mask
        if np.any(green_mask[segmentation]):
            continue  # Skip this mask if it intersects with green areas

        segmented_mask[segmentation] = 1  # Set mask region to 1 if it passes the filters

# plot origin and segmented image(buildings)
plt.imshow(image_rgb)
plt.imshow(segmented_mask, alpha=0.5, cmap='jet')
plt.title("Buildings")
plt.axis('off')
plt.show()

# Store segmentation image
segmented_mask = (segmented_mask * 255).astype(np.uint8)
contours, _ = cv2.findContours(segmented_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
contour_image = image_rgb.copy()
cv2.drawContours(contour_image, contours, -1, (255, 0, 0), 2)
cv2.imwrite("segmented_building.png", cv2.cvtColor(contour_image, cv2.COLOR_RGB2BGR))

# Hiển thị ảnh contours
plt.imshow(contour_image)
plt.title("Segmented Image with Contours")
plt.axis('off')
plt.show()
