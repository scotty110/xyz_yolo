import argparse
import os
import sys
import yaml
from pathlib import Path

import torch
import torch.backends.cudnn as cudnn
import numpy as np

'''
FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative
'''

from yolo.models.common import YoloModel 
#from utils.datasets import IMG_FORMATS, VID_FORMATS, LoadImages, LoadStreams
from yolo.utils.general import (check_img_size, colorstr, cv2, non_max_suppression, scale_coords, xyxy2xywh)
from yolo.utils.augmentations import letterbox
from yolo.utils.plots import Annotator, colors


@torch.no_grad()
def make_model(
        weights='/data/yolov5s.pt',  # model.pt path(s)
        data='/data/coco128.yaml',  # dataset.yaml path
        imgsz=(640, 640),  # inference size (height, width)
        device='cuda:0',  # cuda device, i.e. 0 or 0,1,2,3 or cpu
):
    '''Make and return PyTorch Model (want to have it loaded into memory as frames are fed in'''
    # Load model
    device = torch.device('cuda:0')
    model = YoloModel( weights, device=device, data=data )
    stride, names = model.stride, model.names
    imgsz = check_img_size(imgsz, s=stride)  # check image size

    # Dataloader
    bs = 1  # batch_size
    vid_path, vid_writer = [None] * bs, [None] * bs

    # Run inference
    model.warmup(imgsz=(bs, 3, *imgsz))  # warmup
    return model

def prep_image(img, img_size, stride):
    img0 = cv2.flip(img, 1)  # flip left-right
    # Padded resize
    img = letterbox(img0, img_size, stride=stride)[0]

    # Convert
    img = img.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
    img = np.ascontiguousarray(img)
    return img, img0

@torch.no_grad()
def detect(
        model, # Model loaded previously (want to keep im memory as we wait for imagry)
        image, #Image to process (numpy array)
        data='/data/coco128.yaml',  # dataset.yaml path
        imgsz=(640, 640),  # inference size (height, width)
        conf_thres=0.25,  # confidence threshold
        iou_thres=0.45,  # NMS IOU threshold
        max_det=1000,  # maximum detections per image
        device='cuda:0',  # cuda device, i.e. 0 or 0,1,2,3 or cpu
        nosave=False,  # do not save images/videos
        classes=None,  # filter by class: --class 0, or --class 0 2 3
        agnostic_nms=False,  # class-agnostic NMS
        augment=False,  # augmented inference
        line_thickness=3,  # bounding box thickness (pixels)
        hide_labels=False,  # hide labels
        hide_conf=False,  # hide confidences
):
    ''' We may want to save/return labels in txt format later, but this doesn't seem to be asked for right now TODO'''

    im, im0s = prep_image(image,imgsz,model.stride) # Figure out what this will actually look like
    bs = 1  # batch_size

    # Image Prep
    im = torch.from_numpy(im).to(device)
    im = im.half() if model.fp16 else im.float()  # uint8 to fp16/32
    im /= 255  # 0 - 255 to 0.0 - 1.0
    if len(im.shape) == 3:
        im = im[None]  # expand for batch dim

    # Inference
    pred = model(im, augment=augment )

    # NMS
    pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)

    # Process predictions
    for i, det in enumerate(pred):  # per image
        seen += 1
        # Maybe I don't need you
        #p, im0, frame = path, im0s.copy(), getattr(dataset, 'frame', 0)

        s += '%gx%g ' % im.shape[2:]  # print string
        gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
        annotator = Annotator(im0, line_width=line_thickness, example=str(names))
        if len(det):
            # Rescale boxes from img_size to im0 size
            det[:, :4] = scale_coords(im.shape[2:], det[:, :4], im0.shape).round()

            # Print results
            for c in det[:, -1].unique():
                n = (det[:, -1] == c).sum()  # detections per class
                s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

            # Write results
            for *xyxy, conf, cls in reversed(det):
                # Only going to Produce video for now. This seems easier.
                c = int(cls)  # integer class
                label = None if hide_labels else (names[c] if hide_conf else f'{names[c]} {conf:.2f}')
                annotator.box_label(xyxy, label, color=colors(c, True))

        # Need to return this
        im0 = annotator.result()
    return

@torch.no_grad()
def run(
        weights=ROOT / 'yolov5s.pt',  # model.pt path(s)
        source=ROOT / 'data/images',  # file/dir/URL/glob, 0 for webcam
        data=ROOT / 'data/coco128.yaml',  # dataset.yaml path
        imgsz=(640, 640),  # inference size (height, width)
        conf_thres=0.25,  # confidence threshold
        iou_thres=0.45,  # NMS IOU threshold
        max_det=1000,  # maximum detections per image
        device='cuda:0',  # cuda device, i.e. 0 or 0,1,2,3 or cpu
        nosave=False,  # do not save images/videos
        classes=None,  # filter by class: --class 0, or --class 0 2 3
        agnostic_nms=False,  # class-agnostic NMS
        augment=False,  # augmented inference
        line_thickness=3,  # bounding box thickness (pixels)
        hide_labels=False,  # hide labels
        hide_conf=False,  # hide confidences
):
    # Directories
    ''' We may want to save/return labels in txt format later, but this doesn't seem to be asked for right now TODO'''

    # Load model
    device = torch.device('cuda:0')
    model = YoloModel( weights, device=device, data=data )
    #stride, names, pt = model.stride, model.names, model.pt
    stride, names = model.stride, model.names
    imgsz = check_img_size(imgsz, s=stride)  # check image size

    # Dataloader
    #dataset = LoadImages(source, img_size=imgsz, stride=stride, auto=pt)
    dataset = None #TODO
    bs = 1  # batch_size
    vid_path, vid_writer = [None] * bs, [None] * bs

    # Run inference
    #model.warmup(imgsz=(1 if pt else bs, 3, *imgsz))  # warmup
    model.warmup(imgsz=(bs, 3, *imgsz))  # warmup
    for path, im, im0s, vid_cap, s in dataset:
        
        # Image Prep
        im = torch.from_numpy(im).to(device)
        im = im.half() if model.fp16 else im.float()  # uint8 to fp16/32
        im /= 255  # 0 - 255 to 0.0 - 1.0
        if len(im.shape) == 3:
            im = im[None]  # expand for batch dim

        # Inference
        pred = model(im, augment=augment )

        # NMS
        pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)

        # Process predictions
        for i, det in enumerate(pred):  # per image
            seen += 1
            p, im0, frame = path, im0s.copy(), getattr(dataset, 'frame', 0)

            s += '%gx%g ' % im.shape[2:]  # print string
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            annotator = Annotator(im0, line_width=line_thickness, example=str(names))
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(im.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

                # Write results
                for *xyxy, conf, cls in reversed(det):
                    # Only going to Produce video for now. This seems easier.
                    c = int(cls)  # integer class
                    label = None if hide_labels else (names[c] if hide_conf else f'{names[c]} {conf:.2f}')
                    annotator.box_label(xyxy, label, color=colors(c, True))

            # Stream results
            im0 = annotator.result()
            if view_img:
                cv2.imshow(str(p), im0)
                cv2.waitKey(1)  # 1 millisecond


if __name__ == "__main__":
#    run()
    model = make_model()
    file_name = '/data/teddy_bear.png' 
    img = cv2.imread(file_name)
    print(img.shape)


    detect(model, img)


