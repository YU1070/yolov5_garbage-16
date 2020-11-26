import cv2
import torch
from numpy import random
from models.experimental import attempt_load
from utils.datasets import LoadImages
from utils.general import (check_img_size, non_max_suppression, scale_coords, xyxy2xywh, plot_one_box)
from utils.torch_utils import select_device

print('Setup complete. Using torch %s %s ' % (
    torch.__version__, torch.cuda.get_device_properties(0) if torch.cuda.is_available() else 'CPU'))

int_jpg_path = './inference/images/0.jpg'
out_jpg_path = "./inference/output/0.jpg"
pt_path = './runs/exp0/weights/best.pt'
# Initialize
device = select_device()
# Load model
model = attempt_load(pt_path, map_location=device)  # load FP32 model
imgsz = check_img_size(416, s=model.stride.max())  # check img_size
dataset = LoadImages(int_jpg_path, img_size=imgsz)
# Get names and colors
names = model.module.names if hasattr(model, 'module') else model.names
colors = [[random.randint(0, 255) for _ in range(3)] for _ in range(len(names))]
for path, img, im0s, vid_cap in dataset:
    gn = torch.tensor(im0s.shape)[[1, 0, 1, 0]]  # 获取图片的尺寸
    img = torch.from_numpy(img).to(device)
    img = img.float()  # uint8 to fp16/32
    img /= 255.0  # 0 - 255 to 0.0 - 1.0
    if img.ndimension() == 3:
        img = img.unsqueeze(0)
        # Inference
        pred = model(img)[0]
        # Apply NMS
        pred = non_max_suppression(pred, 0.4, 0.5)
        # Process detections
        for i, det in enumerate(pred):  # detections per image
            # Rescale boxes from img_size to im0 size
            det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0s.shape).round()
            # Write results
            for *xyxy, conf, cls in reversed(det):
                xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                label = '%s %.2f' % (names[int(cls)], conf)
                plot_one_box(xyxy, im0s, label=label, color=colors[int(cls)], line_thickness=3)
    # cv2.imshow("frame", im0s)
    cv2.imwrite(out_jpg_path, im0s)
    if cv2.waitKey(1) & 0xFF == ord('q'): break
cv2.destroyAllWindows()
