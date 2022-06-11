import argparse
import os
import sys
import yaml
from pathlib import Path

import torch
#import torch.backends.cudnn as cudnn
import numpy as np

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

from yolo.models.common import YoloModel 
#from utils.datasets import IMG_FORMATS, VID_FORMATS, LoadImages, LoadStreams
from yolo.utils.general import (check_img_size, colorstr, cv2, non_max_suppression, scale_coords, xyxy2xywh)
from yolo.utils.augmentations import letterbox
from yolo.utils.plots import Annotator, colors

class detector():
    def __init__(self,
        weights = ROOT / 'data/yolov5s.pt',  # model.pt path(s)
        data = ROOT / 'data/coco128.yaml',  # dataset.yaml path
        imgsz =(640, 640),  # inference size (height, width)
        conf_thres = 0.55,  # confidence threshold
        iou_thres = 0.60,  # NMS IOU threshold
        max_det = 1000,  # maximum detections per image
        classes = None,  # filter by class: --class 0, or --class 0 2 3
        agnostic_nms = False,  # class-agnostic NMS
        augment = False,  # augmented inference
        device = None,  # cuda device, i.e. 0 or 0,1,2,3 or cpu
    ):
        # Define which device to use.
        if not device:
            self.device = torch.device('cpu')
            if torch.cuda.is_available():
                self.device = torch.device('cuda:0')
        else:
            self.device = torch.device(device)

        self.precision=torch.float  #
        if torch.cuda.is_available():
            self.precision = torch.half

        # Make model, get class names, and imput image size
        self.imgsz = imgsz
        model_details = self.make_model(
                                        weights = weights,
                                        data = data     
                                        )
        self.model, self.names, self.stride = model_details

        # Others for detections
        self.conf_thres = conf_thres
        self.iou_thres = iou_thres
        self.max_det = max_det
        self.classes = classes
        self.agnostic_nms = agnostic_nms # Not sure what this does -- Look into 
        self.augment = augment
        return None

    @torch.no_grad()
    def make_model(
            self,
            weights,  # model.pt path(s)
            data,  # dataset.yaml path
    ):
        '''Make and return PyTorch Model (want to have it loaded into memory as frames are fed in'''

        # Load model
        model = YoloModel( weights, device=self.device, data=data )
        stride, names = model.stride, model.names
        self.imgsz = check_img_size(self.imgsz, s=stride)  # check image size -- Is this good??

        # Dataloader
        bs = 1  # batch_size
        #vid_path, vid_writer = [None] * bs, [None] * bs

        # Run inference
        model.warmup(imgsz=(bs, 3, *self.imgsz))  # warmup
        return model, names, stride


    def prep_image(self, img):
        img0 = cv2.flip(img, 1)  # flip left-right
        # Padded resize
        img = letterbox(img0, self.imgsz, stride=self.stride)[0]

        # Convert
        img = img.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
        img = np.ascontiguousarray(img)
        return img, img0


    @torch.no_grad()
    def detect(
            self,
            image, # Image to process (numpy array)
            line_thickness=1,  # bounding box thickness (pixels)
            hide_labels=True,  # hide labels
            hide_conf=False,  # hide confidences
    ):
        ''' We may want to save/return labels in txt format later, but this doesn't seem to be asked for right now TODO'''

        im, im0s = self.prep_image(image) # Figure out what this will actually look like
        #bs = 1  # batch_size

        # Image Prep
        im = torch.from_numpy(im).to(self.device)
        #im = im.half() if model.fp16 else im.float()  # uint8 to fp16/32
        im = im.float()
        im /= 255  # 0 - 255 to 0.0 - 1.0
        if len(im.shape) == 3:
            im = im[None]  # expand for batch dim

        # Inference
        pred = self.model(im, augment=self.augment )

        # NMS
        pred = non_max_suppression(pred, self.conf_thres, self.iou_thres, self.classes, self.agnostic_nms, max_det=self.max_det)

        # Process predictions
        for i, det in enumerate(pred):  # per image
            #seen += 1
            # Maybe I don't need you
            #p, im0, frame = path, im0s.copy(), getattr(dataset, 'frame', 0)
            im0 = im0s.copy()

            #s += '%gx%g ' % im.shape[2:]  # print string
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            annotator = Annotator(im0, line_width=line_thickness, example=str(self.names))
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(im.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    #s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

                # Write results
                for *xyxy, conf, cls in reversed(det):
                    # Only going to Produce video for now. This seems easier.
                    c = int(cls)  # integer class
                    label = None if hide_labels else (self.names[c] if hide_conf else f'{self.names[c]} {conf:.2f}')
                    annotator.box_label(xyxy, label, color=colors(c, True))

            # Need to return this
            im0 = annotator.result()
        return im0, pred


if __name__ == "__main__":

    file_name = '/data/street_scene.png' 
    img = cv2.imread(file_name)
    print(img.shape)

    d = detector()
    a_img, _ = d.detect(img)
    cv2.imwrite('/data/annotated.png',a_img)
    print('Done')

