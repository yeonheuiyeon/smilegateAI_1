from __future__ import print_function
import cv2
import numpy as np
import os, errno
from glob import glob

def mkdir_p(path):
    try:
        os.makedirs(path)
    except OSError as exc: #Python > 2.5
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else : raise

def imgColPad(img):
    height, width, channel = img.shape[:3]
    imgPad = np.zeros((height, width*2, channel), np.uint8)
    imgPad[:,:width,:] = img
    return imgPad

def imgColPad_1ch(img):
    height, width = img.shape[:2]
    imgPad = np.zeros((height, width*2), np.uint8)
    imgPad[:,:width] = img
    return imgPad

def Rot90cw(img):
    height, width = img.shape[:2]
    matRot = cv2.getRotationMatrix2D((width/2, height/2), 90, 1)
    imgRotate = cv2.warpAffine(img, matRot, (width,height))
    return imgRotate

def Rot90ccw(img):
    height, width = img.shape[:2]
    matRot = cv2.getRotationMatrix2D((width/2, height/2), -90, 1)
    imgRotate = cv2.warpAffine(img, matRot, (width,height))
    return imgRotate

def get_image_name_list(dir_path):
    jpg_name_list = sorted(glob(os.path.join(dir_path,'*.jpg')))
    png_name_list = sorted(glob(os.path.join(dir_path,'*.png')))
    tif_name_list = sorted(glob(os.path.join(dir_path,'*.tif')))
    pgm_name_list = sorted(glob(os.path.join(dir_path,'*.pgm')))
    image_name_list = jpg_name_list + png_name_list
    return image_name_list

if __name__=='__main__':
    list = get_image_name_list('./data')
    print (list)