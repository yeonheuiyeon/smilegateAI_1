from __future__ import print_function
import cv2
import numpy as np
import os

from . import align_dlib
import dlib

from .tracker import BboxTracker

class FacialFeatureDetector:
    def __init__(self, args):
        ### model init
        self.args = args
        self.bbox = None
        self.landmarks = None
        self.alignedFace = None
        if self.args.flgUseTracker == True:
            self.tracker = BboxTracker(self.args)
        self.flgTracking = False
        
        ### detector initialization
        # dlib
        self.align = align_dlib.AlignDlib(args.dlibFacePredictor, args.dlibCnnDetector, args.detector)
        # ocv old
        self.face_cascade_ocv = cv2.CascadeClassifier('./models/opencv_cascade/haarcascade_frontalface_alt.xml')
        # ocv dnn
        self.dnn_ocv_net = cv2.dnn.readNetFromCaffe(args.ocv_dnn_prototxt, args.ocv_dnn_model)
        # # dlib
        # if self.args.detector == 'hog' or self.args.detector == 'cnn':
        #     self.align = align_dlib.AlignDlib(args.dlibFacePredictor, args.dlibCnnDetector, args.detector)
        # # ocv old
        # elif self.args.detector == 'ocv':
        #     self.face_cascade_ocv = cv2.CascadeClassifier('./models/opencv_cascade/haarcascade_frontalface_alt.xml')
        #     self.align = align_dlib.AlignDlib(args.dlibFacePredictor, args.dlibCnnDetector, args.detector)
        # # ocv dnn
        # elif self.args.detector == 'ssd':
        #     self.dnn_ocv_net = cv2.dnn.readNetFromCaffe(args.ocv_dnn_prototxt, args.ocv_dnn_model)
        #     self.align = align_dlib.AlignDlib(args.dlibFacePredictor, args.dlibCnnDetector, args.detector)
        # else:
        #     self.align = align_dlib.AlignDlib(args.dlibFacePredictor, args.dlibCnnDetector, args.detector)
        #     print ('detector type error: Use [ssd] or [hog] or [cnn]', file=sys.stderr)
        #     exit()
        # for scene change
        self.prev_image = None

    def draw_landmark(self, img, list_points, color=(147,112,219), thickness=-1):
        if self.flgTracking == True:
            color = (127,127,127)
        if list_points == None:
            pass
        for idx, point in enumerate(list_points):
            rad = 1
            cv2.circle(img, point, rad, color, thickness)
    
    def preprocess_for_dlib(self, img):
        bgrImg = img
        if bgrImg is None:
            raise Exception("Unable to load image")
        rgbImg = cv2.cvtColor(bgrImg, cv2.COLOR_BGR2RGB)
        rgbImg_half = cv2.resize(rgbImg, (rgbImg.shape[1]//2, rgbImg.shape[0]//2))
        return rgbImg, rgbImg_half
    
    def detectFace(self, rgbImg):
        bbs, rect_bb = self.align.getLargestFaceBoundingBox(rgbImg)
        
        if rect_bb == None and self.args.flgUseTracker == False:
            return None, None
            self.flgTracking = False
        
        elif rect_bb == None and self.args.flgUseTracker == True:
            ### scene change
            if self.prev_image is not None:
                cimg = rgbImg.astype(np.float32)/255.
                pimg = self.prev_image.astype(np.float32)/255.
                diff = cimg - pimg
                score = abs(np.sum(diff)/(cimg.shape[0]*cimg.shape[1]))
            else:
                score = 0
            
            if score > 0.04:
                self.flgTracking = False
                rect_bb = None
                bbs = None
                self.tracker.tracker = cv2.TrackerKCF_create()
            else:
                bb = self.tracker.updateBbox(rgbImg)
                rect_bb = align_dlib.bb_to_rect(bb)
                if rect_bb is not None:
                    self.flgTracking = True
            
        else:
            self.flgTracking = False
        if rect_bb == None:
            return None, None
        rect_bb_up = dlib.rectangle(left=rect_bb.left()*2, top=rect_bb.top()*2, right=rect_bb.right()*2, bottom=rect_bb.bottom()*2)
        bbox_points = align_dlib.rect_to_bb(rect_bb_up)
        
        self.prev_image = rgbImg
        return rect_bb_up, bbox_points
    
    def detectFace_ocv(self, rgbImg):
        img_gray = cv2.cvtColor(rgbImg, cv2.COLOR_RGB2GRAY)
        bbs, rejectLevels, levelWeights = self.face_cascade_ocv.detectMultiScale3(img_gray, 1.3, 5, outputRejectLevels=True) # x,y,w,h
        if len(bbs) == 0 and self.args.flgUseTracker == False:
            self.flgTracking = False
            return None, None
        elif len(bbs) == 0 and self.args.flgUseTracker == True:
            self.flgTracking = True
            largest_bb_xyxy = self.tracker.updateBbox(rgbImg)
            if largest_bb_xyxy == None:
                return None, None
            x1,y1 = largest_bb_xyxy[0]
            x2,y2 = largest_bb_xyxy[1]
            largest_bb = (x1,y1,x2-x1,y2-y1)
        else:
            largest_bb = max(bbs, key=lambda rect: rect[2]*rect[3])
            self.flgTracking = False
        x,y,w,h = largest_bb
        rect_bb_up = dlib.rectangle(left=2*x,top=2*y,right=2*(x+w),bottom=2*(y+h))
        return rect_bb_up, [(2*x,2*y), (2*(x+w), 2*(y+h))]

    def detectFace_dnn_ocv(self, bgrImg, flgSubDetector=False):
        (h, w) = bgrImg.shape[:2]
        blob = cv2.dnn.blobFromImage(cv2.resize(bgrImg.copy(), (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))
        self.dnn_ocv_net.setInput(blob)
        detections = self.dnn_ocv_net.forward()
        
        bbs = []
        for i in range(0, detections.shape[2]):
            conf_det = detections[0,0,i,2]
            if conf_det < self.args.conf_det:
                continue
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (l, t, r, b) = box.astype("int")
            t_new = int(t+(b-t)*0.1) # remove top margin
            t = t_new
            if (b-t) > (r-l):
                r_new = int((r+l)/2 + (b-t)/2 + 0.5)
                l_new = int((r+l)/2 - (b-t)/2 + 0.5)
                r = r_new
                l = l_new
            x,y,w,h = (l,t,r-l,b-t)
            bbs.append([x,y,w,h])
        
        if flgSubDetector == True:
            if len(bbs) == 0:
                return None, None
            largest_bb = max(bbs, key=lambda rect: rect[2]*rect[3])
            if largest_bb is None:
                return None, None
            self.flgTracking = False
            x,y,w,h = largest_bb

            rect_bb_up = dlib.rectangle(left=x,top=y,right=(x+w),bottom=(y+h))
            return rect_bb_up, [(x,y), ((x+w), (y+h))]

        if len(bbs) == 0 and self.args.flgUseTracker == False:
            self.flgTracking = False
            return None, None

        elif len(bbs) == 0 and self.args.flgUseTracker == True:
            ### scene change
            if self.prev_image is not None:
                cimg = bgrImg.astype(np.float32)/255.
                pimg = self.prev_image.astype(np.float32)/255.
                diff = cimg - pimg
                score = abs(np.sum(diff)/(cimg.shape[0]*cimg.shape[1]))
            else:
                score = 0
            
            if score > 0.04:
                self.flgTracking = False
                largest_bb = None
                bbs = None
                self.tracker.tracker = cv2.TrackerKCF_create()
            else:
                self.flgTracking = True
                largest_bb_xyxy = self.tracker.updateBbox(bgrImg)
                if largest_bb_xyxy == None:
                    return None, None
                x1,y1 = largest_bb_xyxy[0]
                x2,y2 = largest_bb_xyxy[1]
                largest_bb = (x1,y1,x2-x1,y2-y1)
        else:
            largest_bb = max(bbs, key=lambda rect: rect[2]*rect[3])
            self.flgTracking = False
        
        if largest_bb is None:
            return None, []
        
        x,y,w,h = largest_bb

        rect_bb_up = dlib.rectangle(left=x,top=y,right=(x+w),bottom=(y+h))
        return rect_bb_up, [(x,y), ((x+w), (y+h))]

    def detectMultiFace(self, rgbImg):
        rect_bbs, rect_one = self.align.getLargestFaceBoundingBox(rgbImg)
        if rect_one is None:
            return None, None
        list_bb_points = []
        list_rect_bbs = []
        for rect_bb in rect_bbs:
            rect_bb_up = dlib.rectangle(left=rect_bb.left()*2, top=rect_bb.top()*2, right=rect_bb.right()*2, bottom=rect_bb.bottom()*2)
            bbox_points = align_dlib.rect_to_bb(rect_bb_up)
            list_bb_points.append(bbox_points)
            list_rect_bbs.append(rect_bb_up)
        
        return list_rect_bbs, list_bb_points
    
    def detectLandmark(self, rgbImg, rect_bbox):
        alignedFace, landmarks = self.align.align(self.args.dlib_img_size, rgbImg, rect_bbox, landmarkIndices=align_dlib.AlignDlib.OUTER_EYES_AND_NOSE)
        return landmarks, alignedFace

    def refine_bbox(self, bbox, landmarks):
        np_LM = np.array(landmarks)
        bbox_points = [(np.min(np_LM[:,0])-5, np.min(np_LM[:,1])-10), (np.max(np_LM[:,0])+5,np.max(np_LM[:,1])+5)]
        return bbox_points

    def refine_bboxes(self, list_face_candidate):
        if list_face_candidate is None:
            return None
        for face_candidate in list_face_candidate:
            if len(face_candidate) == 2 and face_candidate[1] != None:
                bbox_point = self.refine_bbox(face_candidate[0], face_candidate[1])
                face_candidate[0] = bbox_point
        return list_face_candidate
    
    def detectFF(self, img):
        self.bbox = None
        self.landmarks = None
        self.alignedFace = None
        
        # start = cv2.getTickCount()
        self.bgrImg = img
        self.rgbImg, rgbImg_half = self.preprocess_for_dlib(img)
        if self.args.detector == 'hog' or self.args.detector == 'cnn':
            rect_bb, bbox_points_origin = self.detectFace(rgbImg_half)
            if rect_bb is None:
                rect_bb, bbox_points_origin = self.detectFace_dnn_ocv(self.bgrImg, flgSubDetector=True)
        elif self.args.detector == 'ocv':
            rect_bb, bbox_points_origin = self.detectFace_ocv(rgbImg_half)
        elif self.args.detector == 'ssd':
            rect_bb, bbox_points_origin = self.detectFace_dnn_ocv(self.bgrImg)
        self.bbox = bbox_points_origin
        # detection_time = (cv2.getTickCount()-start)/cv2.getTickFrequency() * 1000
        # print ('detection time: %.2fms'%detection_time)
        if rect_bb == None:
            return None, None, None
        
        landmarks, alignedFace = self.detectLandmark(self.rgbImg, rect_bb)
        
        if landmarks == None:
            return self.bbox, None, None
        
        if self.args.flgUseTracker == True and self.flgTracking == True:
            refined_bbox_points = bbox_points_origin
        else:
            refined_bbox_points = self.refine_bbox(bbox_points_origin, landmarks)

        if self.args.flgUseTracker == True and self.flgTracking == False:
            bbox_xywh = self.tracker.xyxy2xywh(refined_bbox_points)
            if self.args.detector == 'ssd':
                bbox_xyxy = self.tracker.xywh2xyxy(bbox_xywh)
                ok = self.tracker.initTracker(rgbImg_half, bbox_xyxy)

            else:
                bbox_xywh_half = tuple([i//2 for i in bbox_xywh])
                bbox_xyxy_half = self.tracker.xywh2xyxy(bbox_xywh_half)
                ok = self.tracker.initTracker(self.bgrImg, bbox_xyxy_half)
        
        self.bbox = refined_bbox_points
        self.landmarks = landmarks
        self.alignedFace = alignedFace
        return self.bbox, self.landmarks, self.alignedFace

    def detect_multiFacialFeature(self, img):
        self.list_bboxes = None
        self.list_landmarks = None
        self.alignedFace = None
        
        self.bgrImg = img
        self.rgbImg, rgbImg_half = self.preprocess_for_dlib(img)
        list_rect_bbs, list_bbox_points_origin = self.detectMultiFace(rgbImg_half)
        self.list_bboxes = list_bbox_points_origin
        if list_rect_bbs == None:
            return None, None, None
        self.list_landmarks = []
        self.list_face = []
        for rect_bb, bbox_points_origin in zip(list_rect_bbs, list_bbox_points_origin):
            landmarks, alignedFace = self.detectLandmark(self.rgbImg, rect_bb)
            if landmarks == None:
                landmarks = [-1]
            # self.list_landmarks.append(landmarks)
            self.list_face.append([bbox_points_origin, landmarks])

        list_refined_face = self.refine_bboxes(list_face_candidate=self.list_face)

        # if self.args.flgUseTracker == True and self.flgTracking == True:
        #     refined_bbox_points = bbox_points_origin
        # else:
        #     refined_bbox_points = self.refine_bbox(bbox_points_origin, landmarks)

        # if self.args.flgUseTracker == True and self.flgTracking == False:
        #     bbox_xywh = self.tracker.xyxy2xywh(refined_bbox_points)
        #     bbox_xywh_half = tuple([i/2 for i in bbox_xywh])
        #     bbox_xyxy_half = self.tracker.xywh2xyxy(bbox_xywh_half)
        #     ok = self.tracker.initTracker(rgbImg_half, bbox_xyxy_half)
        
        np_refined_face = np.array(list_refined_face)
        np_refined_face = np.swapaxes(np_refined_face, 0, 1)
        # print np_refined_face[0].tolist()
        self.list_bboxes = np_refined_face[0].tolist()
        self.list_landmarks = np_refined_face[1].tolist()
        
        # self.list_bboxes = list_refined_face
        # self.landmarks = list_landmarks
        # self.alignedFace = alignedFace
        return self.list_bboxes, self.list_landmarks, self.alignedFace

    def drawFacialFeatures(self, show, bbox=None, landmarks=None):
        if self.flgTracking == False:
            # color_rect = (128,255,0)
            color_rect = (145,214,0)
        else:
            color_rect = (0,0,255)
        if bbox != None:
            cv2.rectangle(show, bbox[0], bbox[1], color_rect, 2)
        elif self.bbox != None:
            cv2.rectangle(show, self.bbox[0], self.bbox[1], color_rect, 2)
        if landmarks != None:
            self.draw_landmark(show, landmarks)
        elif self.landmarks != None:
            self.draw_landmark(show, self.landmarks)
    
    def analyze_features(self, bbox=None, landmarks=None):
        if bbox == None:
            if self.bbox != None:
                bbox = self.bbox
            else:
                return -1
        bbox_ratio = (bbox[1][0]-bbox[0][0])/float(bbox[1][1]-bbox[0][1])
        return bbox_ratio

    def drawMultiFFs(self, show, list_bboxes=None, list_landmarks=None):
        if self.flgTracking == False:
            color_rect = (255,128,0)
        else:
            color_rect = (0,0,255)
        if list_bboxes != None:
            for bbox in list_bboxes:
                cv2.rectangle(show, bbox[0], bbox[1], color_rect, 2)
        elif self.list_bboxes != None:
            for bbox in self.list_bboxes:
                cv2.rectangle(show, bbox[0], bbox[1], color_rect, 2)
        if list_landmarks != None:
            for landmarks in list_landmarks:
                if len(landmarks) == 1:
                    continue
                self.draw_landmark(show, landmarks)
        elif self.list_landmarks != None:
            for landmarks in self.list_landmarks:
                if len(landmarks) == 1:
                    continue
                self.draw_landmark(show, landmarks)

if __name__ == '__main__':
    class args:
        def __init__(self):
            self.detector = 'ssd'
            self.ocv_dnn_prototxt = '../models/cv_dnn/deploy.prototxt.txt'
            self.ocv_dnn_model = '../models/cv_dnn/res10_300x300_ssd_iter_140000.caffemodel'
            self.dlibFacePredictor = '../models/dlib/shape_predictor_68_face_landmarks.dat'
            self.dlibCnnDetector = '../models/dlib/mmod_human_face_detector.dat'
            self.flgUseTracker = False
            self.conf_det = 0.6
            self.dlib_img_size = 128
    flag = args()
    
    objFFD = FacialFeatureDetector(flag)

    vc = cv2.VideoCapture('../data/tkwoo_20180328.mp4')

    while True:
        _, bgr_image = vc.read()
        bbox, landmarks, alignedFace = objFFD.detectFF(bgr_image)
