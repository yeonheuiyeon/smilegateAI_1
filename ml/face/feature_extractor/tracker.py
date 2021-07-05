import cv2
import os
import numpy as np

class BboxTracker():
    def __init__(self, args):
        self.args = args
        # self.tracker = cv2.TrackerMedianFlow_create()
        self.tracker = cv2.TrackerKCF_create()
    
    def xyxy2xywh(self, bbox_xyxy):
        x = bbox_xyxy[0][0]
        y = bbox_xyxy[0][1]
        w = bbox_xyxy[1][0] - bbox_xyxy[0][0]
        h = bbox_xyxy[1][1] - bbox_xyxy[0][1]
        return (x,y,w,h)

    def xywh2xyxy(self, bbox_xywh):
        x1 = int(bbox_xywh[0])
        y1 = int(bbox_xywh[1])
        x2 = int(bbox_xywh[0] + bbox_xywh[2])
        y2 = int(bbox_xywh[1] + bbox_xywh[3])
        return [(x1,y1),(x2,y2)]

    def initTracker(self, img, bbox):
        bbox_xywh = self.xyxy2xywh(bbox)
        self.tracker = cv2.TrackerKCF_create()
        # self.tracker = cv2.TrackerMedianFlow_create()
        ok = self.tracker.init(img, bbox_xywh)
        return ok
    
    def updateBbox(self, img):
        ok, bbox_xywh = self.tracker.update(img)
        if ok:
            return self.xywh2xyxy(bbox_xywh)
        else:
            return None
