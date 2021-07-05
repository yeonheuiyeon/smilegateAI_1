from __future__ import print_function
import argparse
import cv2
import os
import numpy as np
from random import shuffle

from feature_extractor import utils
from feature_extractor.detector import FacialFeatureDetector
from pose_estimation.eye_gaze import GazeEstimator
from pose_estimation.head_pose import HeadPoseEstimator

CUR_PWD=os.path.dirname(os.path.abspath(__file__)) 

np.set_printoptions(precision=2)

### Model path
fileDir = os.path.dirname(os.path.realpath(__file__))
modelDir = os.path.join(fileDir, 'models')
dlibModelDir = os.path.join(modelDir, 'dlib')

### arg parser
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('imgs', type=str, help="Input images.") #nargs='+', 

    ### detector args
    parser.add_argument('--detector', type=str, help='[hog], [cnn], [ssd], [ocv]', default='hog')
    parser.add_argument('--dlibFacePredictor', type=str, help="Path to dlib's face predictor.",
                        default=os.path.join(dlibModelDir, "shape_predictor_68_face_landmarks.dat"))
    parser.add_argument('--dlibCnnDetector', type=str, help="Path to dlib's face predictor.",
                        default=os.path.join(dlibModelDir, "mmod_human_face_detector.dat"))
    parser.add_argument('--dlib_img_size', type=int,
                        help="Default image dimension.", default=128)
    parser.add_argument("--ocv_dnn_prototxt", help="path to Caffe 'deploy' prototxt file for ocv dnn detector", default='{}/models/cv_dnn/deploy.prototxt.txt'.format(CUR_PWD))
    parser.add_argument("--ocv_dnn_model", help="path to Caffe pre-trained model", default='{}/models/cv_dnn/res10_300x300_ssd_iter_140000.caffemodel'.format(CUR_PWD))
    parser.add_argument("--conf_det", type=float, default=0.6, help="minimum probability to filter weak detections for ocv dnn detector")
    parser.add_argument('--flgUseTracker', type=bool, default=True)
    parser.add_argument('--flgFlip', type=str, default='False')
    parser.add_argument('--flgMultiTarget', type=str, default='False')

    ### classifier args
    parser.add_argument("--out_saveImgSeq", default=None, type=str)
    parser.add_argument("--extend_show_img", default='False', type=str)
    parser.add_argument("--icon_path", default='{}/vis'.format(CUR_PWD), type=str)

    parser.add_argument("--tf_log_level", help="0, 1, 2, 3", default='2', type=str)

    ### output
    parser.add_argument("--out_video_name", default=None)

    args = parser.parse_args()
    return args


args = parse_args()

os.environ['TF_CPP_MIN_LOG_LEVEL'] = args.tf_log_level # or any {'0', '1', '2', '3'}


def videoFD(vc):
    # cv2.namedWindow('skincolor',0)
    # cv2.resizeWindow('skincolor', 400, 400)
    cv2.namedWindow('show',0)
    cv2.resizeWindow('show', 960, 720)
    
    ### initialize
    objFFD = FacialFeatureDetector(args)
    objGE = GazeEstimator(args)
    objHPE = HeadPoseEstimator(args)
    
    vw = set_videowriter(args, vc)
        
    flgSkip=False
    idx_frame = 0
    while True:
        bgr_image = vc.read()[1]
        if args.flgFlip == 'True':
            bgr_image = cv2.flip(bgr_image, 1)
        if type(bgr_image) != np.ndarray:
            print ('no frame')
            continue
        if bgr_image.shape[1] > 1000 or bgr_image.shape[0] > 1000:
            bgr_image = cv2.resize(bgr_image, (bgr_image.shape[1]//2, bgr_image.shape[0]//2))
        elif bgr_image.shape[1]*bgr_image.shape[0] > 300000:
            flgSkip = not flgSkip
            if flgSkip == True:
                continue
        
        ### module processing
        start = cv2.getTickCount()

        bbox, landmarks, alignedFace = objFFD.detectFF(bgr_image)
        eye_centers = objGE.detectPupil(bgr_image, landmarks, objFFD.flgTracking)
        rotation, rvec, tvec, nose_point = objHPE.estimateHeadPose(bgr_image, landmarks, objFFD.flgTracking)

        bbox_ratio = objFFD.analyze_features()

        time = (cv2.getTickCount() - start)/cv2.getTickFrequency()
        
        # print (rotation)
        # print ('%s, %.3f, %s, %s'%(str(bgr_image.shape[:2]), bbox_ratio, objFFD.flgTracking, str_sm_emotion))
        # print (eye_centers)
        print ('time: %.2fms'%(time*1000))
        
        ### draw image and results
        show = bgr_image.copy()
        cv2.putText(show, '%.1ffps'%(1/time) , (show.shape[1]-65,15), cv2.FONT_HERSHEY_DUPLEX, 0.5, (127,127,127))
        objFFD.drawFacialFeatures(show)
        objGE.draw_pupil(show, eye_centers)
        objHPE.draw_pose(show, rotation, rvec, tvec, nose_point, unit_size=300.)

        state = cv_show(show=show, args=args, vw=vw, wait_time=1)
        if state == -1:
            break
        
        idx_frame += 1


def main():
    ### detect & landmark

    ext = os.path.splitext(args.imgs)[-1]
    
    if ext == '.mp4':
        vc = cv2.VideoCapture(args.imgs)
        videoFD(vc)
    elif args.imgs == '-' or args.imgs == 'webcam':
        vc = cv2.VideoCapture(0+ cv2.CAP_DSHOW)
        videoFD(vc)
    elif os.path.isdir(args.imgs):
        print ('not supported yet')
        return
    else:
        vc = cv2.VideoCapture(0+ cv2.CAP_DSHOW)
        videoFD(vc)


def cv_show(show, args, vw, wait_time=1):
    cv2.imshow("show", show)
    if args.out_video_name is not None:
        vw.write(show)
    key = cv2.waitKey(wait_time)
    if key == 27:
        # vw.release()
        return -1
    elif key & 0xFF == ord('p'):
        cv2.waitKey()
    if args.out_saveImgSeq != None and os.path.exists(args.out_saveImgSeq):
        cv2.imwrite(os.path.join(args.out_saveImgSeq, '%010d.png'%idx_frame), show)
    return 0


def set_videowriter(args, vc):
    ### video writer initialize
    if args.out_video_name is not None:
        bgr_image = vc.read()[1]
        if bgr_image.shape[1] > 1000 or bgr_image.shape[0] > 1000:
            bgr_image = cv2.resize(bgr_image, (bgr_image.shape[1]//2, bgr_image.shape[0]//2))
        # fourcc = cv2.VideoWriter_fourcc(*'avc1')
        # fourcc = cv2.VideoWriter_fourcc(*'x264')
        fourcc = cv2.VideoWriter_fourcc(*'MP4V')
        # fourcc = cv2.VideoWriter_fourcc(*'MJPG')
        # vw = cv2.VideoWriter(args.out_video_name, fourcc, 60.0, (bgr_image.shape[1], bgr_image.shape[0]))
        video_width = bgr_image.shape[1]
        if args.extend_show_img == 'True' and args.flgMultiTarget == 'False':
            video_width = bgr_image.shape[1] + 100
        vw = cv2.VideoWriter(args.out_video_name, fourcc, 30.0, (video_width, bgr_image.shape[0]))
        # vw = cv2.VideoWriter(args.out_video_name, fourcc, 30.0, (640, 300))
    else:
        vw = None
    return vw


if __name__=='__main__':
    main()