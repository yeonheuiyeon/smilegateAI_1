from __future__ import print_function

import cv2
import numpy as np
import math


class HeadPoseEstimator:
    def __init__(self, flag):
        self.flag = flag
        # 3D model points.
        self.model_points = np.array([
                            (0.0, 0.0, 0.0),             # Nose tip 30
                            (0.0, -330.0, -65.0),        # Chin 8
                            (-225.0, 170.0, -135.0),     # Left eye left corner 36
                            (225.0, 170.0, -135.0),      # Right eye right corne 45
                            (-150.0, -150.0, -125.0),    # Left Mouth corner 48
                            (150.0, -150.0, -125.0)      # Right mouth corner 54
                        ])
    def estimateMultiHeadPose(self, bgr_img, list_landmark):
        if list_landmark is None:
            return
        list_extrinsic_param = []
        for landmark in list_landmark:
            extrinsic_param = self.estimateHeadPose(bgr_img, landmark)
            if extrinsic_param[0] is not None:
                list_extrinsic_param.append(extrinsic_param)
        return list_extrinsic_param
        
    def estimateHeadPose(self, bgr_img, landmark, flgTracking=False):
        if landmark is None or flgTracking == True:
            return None, None, None, None
        # camera parameter setting
        focal_length = bgr_img.shape[1]
        center = (bgr_img.shape[1]/2, bgr_img.shape[0]/2)
        self.camera_matrix = np.array(
                                 [[focal_length, 0, center[0]],
                                 [0, focal_length, center[1]],
                                 [0, 0, 1]], dtype='double'
                                )
        self.dist_coeffs = np.zeros((4,1)) # distortion coefficent
        
        # image point setting
        image_points = np.array([
                                 landmark[30],
                                 landmark[8],
                                 landmark[36],
                                 landmark[45],
                                 landmark[48],
                                 landmark[54]
                                ], dtype='double')

        # 3D, 2D point to extrinsic matrix
        (success, rot_vec, tr_vec) = cv2.solvePnP(self.model_points, 
                                                  image_points,
                                                  self.camera_matrix,
                                                  self.dist_coeffs,
                                                  flags=cv2.SOLVEPNP_ITERATIVE
                                                 )
        
        # degree translation
        R, _ = cv2.Rodrigues(rot_vec)
        EulerAngle = self.rotationMatrixToEulerAngles(R)
        rotation = EulerAngle*180/3.141592 # rotation : pitch, yaw, roll
        
        rotation[0] = (180-abs(rotation[0]))*(rotation[0]/abs(rotation[0]))
        # print ('rotation:', rotation)

        return [rotation, rot_vec, tr_vec, image_points]

    def draw_multipose(self, show, list_extrinsic_param, unit_size=500.):
        if list_extrinsic_param is None:
            return
        elif len(list_extrinsic_param) == 0:
            return

        for extrinsic_param in list_extrinsic_param:
            self.draw_pose(show, extrinsic_param[0], extrinsic_param[1],
                           extrinsic_param[2], extrinsic_param[3],
                           unit_size)

    def draw_pose(self, show, rotation, rvec, tvec, image_points, unit_size = 500.):
        if rotation is None:
            return
        elif abs(rotation[0]) > 25 or abs(rotation[1]) > 55 or abs(rotation[2]) > 35: 
            return
            
        (nose_end_point2D_x, jacobian) = cv2.projectPoints(np.array([(unit_size, 0.0, 0.0)]), rvec, tvec, self.camera_matrix, self.dist_coeffs)
        (nose_end_point2D_y, jacobian) = cv2.projectPoints(np.array([(0.0, unit_size, 0.0)]), rvec, tvec, self.camera_matrix, self.dist_coeffs)
        (nose_end_point2D_z, jacobian) = cv2.projectPoints(np.array([(0.0, 0.0, unit_size)]), rvec, tvec, self.camera_matrix, self.dist_coeffs)
        
        # for p in image_points:
        #     cv2.circle(bgr_img, (int(p[0]),int(p[1])), 3, (0,0,255), -1)

        p1 = (int(image_points[0][0]), int(image_points[0][1]))
        p2 = (int(nose_end_point2D_z[0][0][0]), int(nose_end_point2D_z[0][0][1]))
        p3 = (int(nose_end_point2D_x[0][0][0]), int(nose_end_point2D_x[0][0][1]))
        p4 = (int(nose_end_point2D_y[0][0][0]), int(nose_end_point2D_y[0][0][1]))

        cv2.line(show, p1, p3, (0,0,255), 2)
        cv2.line(show, p1, p4, (0,255,0), 2)
        cv2.line(show, p1, p2, (255,0,0), 2)

    def isRotationMatrix(self, R):
        Rt = np.transpose(R)
        shouldBeIdentity = np.dot(Rt, R)
        I = np.identity(3, dtype = R.dtype)
        n = np.linalg.norm(I - shouldBeIdentity)
        return n < 1e-6

    def rotationMatrixToEulerAngles(self, R): 
        assert(self.isRotationMatrix(R))
        sy = math.sqrt(R[0,0] * R[0,0] +  R[1,0] * R[1,0])
        singular = sy < 1e-6
        if  not singular :
            x = math.atan2(R[2,1] , R[2,2])
            y = math.atan2(-R[2,0], sy)
            z = math.atan2(R[1,0], R[0,0])
        else :
            x = math.atan2(-R[1,2], R[1,1])
            y = math.atan2(-R[2,0], sy)
            z = 0
        return np.array([x, y, z])


