import torch
import cv2 as cv
import numpy as np
from numpy import random
from PIL import ImageGrab
from models.experimental import attempt_load
from utils.general import (non_max_suppression, scale_coords, xyxy2xywh, plot_one_box)
from utils.torch_utils import select_device, load_classifier, time_synchronized

print('Setup complete. Using torch %s %s ' % (
    torch.__version__, torch.cuda.get_device_properties(0) if torch.cuda.is_available() else 'CPU'))
# Initialize
device = select_device()
# Load model
model = attempt_load('yolov5s.pt', map_location=device)  # load FP32 model
# Get names and colors
names = model.module.names if hasattr(model, 'module') else model.names
colors = [[random.randint(0, 255) for _ in range(3)] for _ in range(len(names))]


def process_img(original_image):
    processde_img = cv.cvtColor(original_image, cv.COLOR_BGR2RGB)
    processde_img = cv.resize(processde_img, (416, 352))
    return processde_img


while (True):
    printscreen_pil = np.array(ImageGrab.grab(bbox=(0, 40, 800, 800)))
    frame = process_img(printscreen_pil)
    img = frame.copy()
    img = np.transpose(img, (2, 0, 1))
    img = torch.from_numpy(img).to(device)
    img = img.float()  # uint8 to fp16/32
    img /= 255.0  # 0 - 255 to 0.0 - 1.0
    if img.ndimension() == 3:
        img = img.unsqueeze(0)
    # Inference
    pred = model(img)[0]
    # Apply NMS
    pred = non_max_suppression(pred, 0.4, 0.5)
    gn = torch.tensor(frame.shape)[[1, 0, 1, 0]]
    if pred != [None]:
        # Process detections
        for i, det in enumerate(pred):  # detections per image
            # Rescale boxes from img_size to im0 size
            det[:, :4] = scale_coords(img.shape[2:], det[:, :4], frame.shape).round()
            # Write results
            for *xyxy, conf, cls in reversed(det):
                xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                label = '%s %.2f' % (names[int(cls)], conf)
                plot_one_box(xyxy, frame, label=label, color=colors[int(cls)], line_thickness=3)
    cv.imshow("frame", frame)
    if cv.waitKey(1) & 0xFF == ord('q'): break
cv.destroyAllWindows()
