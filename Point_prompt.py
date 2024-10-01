import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt
from segment_anything import SamPredictor, sam_model_registry, SamAutomaticMaskGenerator
from groundingdino.util.inference import load_model, load_image, predict
from sentinelhub import SHConfig, SentinelHubRequest, MimeType, CRS, DataCollection, Geometry
'''
# Cấu hình thông tin truy cập Sentinel Hub
config = SHConfig()
config.sh_client_id = 'f8b2e7a0-62d9-49f3-9cd2-7143ed1c04c7'  # Thay bằng Client ID của bạn
config.sh_client_secret = 'LwIomvPcB8e9fhengRlcQRKU0US5ggFT'  # Thay bằng Client Secret của bạn

# Định nghĩa polygon từ GeoJSON
polygon_geojson = {
    "type": "Polygon",
    "coordinates": [[[105.805206, 20.989171], [105.763321, 20.989171], [105.763321, 21.032891], [105.805206, 21.032891], [105.805206, 20.989171]]]
}
polygon = Geometry(polygon_geojson, CRS.WGS84)

# Tạo yêu cầu dữ liệu từ Sentinel-2
request = SentinelHubRequest(
    evalscript="""
        // Sử dụng kênh B04, B03, B02 cho Sentinel-2 L2A
        return [B04, B03, B02];
    """,
    input_data=[
        SentinelHubRequest.input_data(
            data_collection=DataCollection.SENTINEL2_L2A,
            time_interval=('2024-09-01', '2024-09-24'),
        )
    ],
    responses=[
        SentinelHubRequest.output_response('default', MimeType.PNG)
    ],
    geometry=polygon,  # Sử dụng polygon thay cho bbox
    size=(512, 512),
    config=config
)

# Gửi yêu cầu và lấy ảnh
image = request.get_data()[0]

# Hiển thị ảnh bằng matplotlib
plt.imshow(image)
plt.title("Sentinel-2 Image")
plt.axis('off')
plt.show()
'''
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

# Bước 3: Dự đoán mask từ các điểm
predictor.set_image(image_rgb)
input_point = np.array([[945,373],[949,122],[866,650]])  # Điều chỉnh tùy thuộc vào ảnh
input_label = np.array([1,1,1])#1 foreground 0 background
masks, _, _ = predictor.predict(point_coords=input_point, point_labels=input_label)#dự đoán 1 điểm với 1 shot
segmented_mask = masks[0]
# Hiển thị kết quả
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.imshow(image_rgb)
plt.title("Original Image")
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(image_rgb)
plt.imshow(segmented_mask, alpha=0.5, cmap='jet')
plt.title("Segmented Building")
plt.axis('off')
plt.show()

# Lưu kết quả segmentation
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
