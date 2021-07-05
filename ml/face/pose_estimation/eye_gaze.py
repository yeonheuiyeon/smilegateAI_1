from __future__ import print_function
import cv2
import numpy as np
# import matplotlib.pyplot as plt


class GazeEstimator:
    def __init__(self, flag):
        self.flag = flag
        pass

    def eye_crop(self, bgr_img, landmark):
        # dlib eye landmark: 36~41 (6), 42~47 (6)
        np_left_eye_points = np.array(landmark[36:42])
        np_right_eye_points = np.array(landmark[42:48])

        np_left_tl = np_left_eye_points.min(axis=0)
        np_left_br = np_left_eye_points.max(axis=0)
        np_right_tl = np_right_eye_points.min(axis=0)
        np_right_br = np_right_eye_points.max(axis=0)

        list_left_tl = np_left_tl.tolist()
        list_left_br = np_left_br.tolist()
        list_right_tl = np_right_tl.tolist()
        list_right_br = np_right_br.tolist()
        
        # print (np_left_br - np_left_tl)
        self.left_eye_size = np_left_br - np_left_tl
        self.right_eye_size = np_right_br - np_right_tl
        
        ### if eye size is small
        if self.left_eye_size[1] < 5:
            margin = 1
        else:
            margin = 6
        
        img_left_eye = bgr_img[np_left_tl[1]-margin:np_left_br[1]+margin, np_left_tl[0]-margin//2:np_left_br[0]+margin//2]
        img_right_eye = bgr_img[np_right_tl[1]-margin:np_right_br[1]+margin, np_right_tl[0]-margin//2:np_right_br[0]+margin//2]

        return [img_left_eye, img_right_eye]
    
    def findCenterPoint(self, gray_eye, str_direction='left'):
        if gray_eye is None:
            return [0, 0]
        filtered_eye = cv2.bilateralFilter(gray_eye, 7, 75, 75)
        filtered_eye = cv2.bilateralFilter(filtered_eye, 7, 75, 75)
        filtered_eye = cv2.bilateralFilter(filtered_eye, 7, 75, 75)

        # 2D images -> 1D signals
        row_sum = 255 - np.sum(filtered_eye, axis=0)//gray_eye.shape[0]
        col_sum = 255 - np.sum(filtered_eye, axis=1)//gray_eye.shape[1]

        # normalization & stabilization
        def vector_normalization(vector):
            vector = vector.astype(np.float32)
            vector = (vector-vector.min())/(vector.max()-vector.min()+1e-6)*255
            vector = vector.astype(np.uint8)
            vector = cv2.blur(vector, (5,1)).reshape((vector.shape[0],))
            vector = cv2.blur(vector, (5,1)).reshape((vector.shape[0],))            
            return vector
        row_sum = vector_normalization(row_sum)
        col_sum = vector_normalization(col_sum)

        def findOptimalCenter(gray_eye, vector, str_axis='x'):
            axis = 1 if str_axis == 'x' else 0
            center_from_start = np.argmax(vector)
            center_from_end = gray_eye.shape[axis]-1 - np.argmax(np.flip(vector,axis=0))
            return (center_from_end + center_from_start) // 2

        # New
        center_x = findOptimalCenter(gray_eye, row_sum, 'x')
        center_y = findOptimalCenter(gray_eye, col_sum, 'y')

        

        inv_eye = (255 - filtered_eye).astype(np.float32)
        inv_eye = (255*(inv_eye - inv_eye.min())/(inv_eye.max()-inv_eye.min())).astype(np.uint8)

        
        resized_inv_eye = cv2.resize(inv_eye, (inv_eye.shape[1]//3, inv_eye.shape[0]//3))
        init_point = np.unravel_index(np.argmax(resized_inv_eye),resized_inv_eye.shape)

        x_candidate = init_point[1]*3 + 1
        for idx in range(10):
            temp_sum = row_sum[x_candidate-2:x_candidate+3].sum()
            if temp_sum == 0:
                break
            normalized_row_sum_part = row_sum[x_candidate-2:x_candidate+3].astype(np.float32)//temp_sum
            moving_factor = normalized_row_sum_part[3:5].sum() - normalized_row_sum_part[0:2].sum()
            if moving_factor > 0.0:
                x_candidate += 1
            elif moving_factor < 0.0:
                x_candidate -= 1
        
        center_x = x_candidate

        if center_x >= gray_eye.shape[1]-2 or center_x <= 2:
            # color = color_not_find
            center_x = -1
        elif center_y >= gray_eye.shape[0]-1 or center_y <= 1:
            center_y = -1
        
        return [center_x, center_y]
    
    def detectMultiPupil(self, bgr_img, list_landmark):
        if list_landmark is None:
            return
        
        list_centers = []
        for landmark in list_landmark:
            centers = self.detectPupil(bgr_img, landmark, False)
            if centers is not None:
                list_centers.append(centers)
        
        return list_centers

    def detectPupil(self, bgr_img, landmark, flgTracking):
        if landmark is None or flgTracking==True:
            return
        self.landmark = landmark
        self.left_eye_size = self.right_eye_size = [0,0]
        self.img_eyes = []
        self.img_eyes = self.eye_crop(bgr_img, landmark)

        gray_left_eye = cv2.cvtColor(self.img_eyes[0], cv2.COLOR_BGR2GRAY)
        gray_right_eye = cv2.cvtColor(self.img_eyes[1], cv2.COLOR_BGR2GRAY)

        if gray_left_eye is None or gray_right_eye is None:
            return 

        left_center_x, left_center_y = self.findCenterPoint(gray_left_eye,'left')
        right_center_x, right_center_y = self.findCenterPoint(gray_right_eye,'right')

       
        return [left_center_x, left_center_y, right_center_x, right_center_y, gray_left_eye.shape, gray_right_eye.shape]

    def draw_multipupil(self, show, list_landmark, list_centers):
        if list_centers is None:
            return
        elif len(list_centers) == 0:
            return

        for idx, landmark in enumerate(list_landmark):
            show_eyes = self.eye_crop(show, landmark)
            centers = list_centers[idx]
            self.draw_eye_center(show_eyes[0], centers[0], centers[1], 'left')
            self.draw_eye_center(show_eyes[1], centers[2], centers[3], 'right')

    def draw_pupil(self, show, centers):
        if centers is None:
            return
        show_eyes = self.eye_crop(show, self.landmark)
        self.draw_eye_center(show_eyes[0], centers[0], centers[1], 'left')
        self.draw_eye_center(show_eyes[1], centers[2], centers[3], 'right')

    def draw_eye_center(self, eye_img, center_x, center_y, str_direction='left'):
        if eye_img is None or center_x is None or center_y is None:
            return
        # color_find = (3,107,255)
        color_find = (77,202,240)
        color_not_find = (50,50,50)
        
        # eye_size = self.left_eye_size if str_direction == 'left' else self.right_eye_size
        eye_size = [eye_img.shape[1] - 6, eye_img.shape[0] - 12]
        
        color = color_find if eye_size[1] > 6 else color_not_find
        if center_x >= eye_img.shape[1]-2 or center_x <= 2:
            color = color_not_find
        elif center_y >= eye_img.shape[0]-1 or center_y <= 1:
            color = color_not_find

        sz_CH = 3 # size of crosshair
        cv2.line(eye_img, (center_x-sz_CH,center_y), (center_x+sz_CH,center_y), color=color, thickness=2)
        cv2.line(eye_img, (center_x,center_y-sz_CH), (center_x,center_y+sz_CH), color=color, thickness=2)

    def findSkinMask(self, hsv_img):
        lower = np.array([0,30,70], dtype='uint8')
        upper = np.array([20,255,255], dtype='uint8')
        lower_red = np.array([175,30,70], dtype='uint8')
        upper_red = np.array([180,255,255], dtype='uint8')
        skinMask = cv2.inRange(hsv_img, lower, upper)
        skinMask2 = cv2.inRange(hsv_img, lower_red, upper_red)
        skinMask = cv2.bitwise_or(skinMask, skinMask2)
        _, skinMask = cv2.threshold(skinMask, 128, 128, cv2.THRESH_BINARY)
        return skinMask
    