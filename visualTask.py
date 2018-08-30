## ---------------------------------------------------------------------
# author: Meringue
# date: 1/15/2018
# description: visual classes for Nao golf task.
## ---------------------------------------------------------------------
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import os
sys.path.append("/home/meringue/Softwares/pynaoqi-sdk/") #naoqi directory
sys.path.append("./")
#os.chdir(os.getcwd())
import cv2
import numpy as np

import vision_definitions as vd
import time

from configureNao import ConfigureNao
from naoqi import ALProxy
sys.path.append("/home/meringue/Documents/nao-object-detection/yoloNet")
from yolo.net.yolo_tiny_net import YoloTinyNet
import tensorflow as tf


class VisualBasis(ConfigureNao):
    """
    a basic class for visual task.
    """
    
    def __init__(self, IP, cameraId, resolution=vd.kVGA):
        """
        initilization.
        
        Arguments:
        IP -- NAO's IP
        cameraId -- bottom camera (1,default) or top camera (0).
        resolution -- (kVGA, default: 640*480)
        """  
        
        super(VisualBasis, self).__init__(IP)
        self._cameraId = cameraId
        self._resolution = resolution
        
        self._colorSpace = vd.kBGRColorSpace
        self._fps = 20

        self._frameHeight = 0
        self._frameWidth = 0
        self._frameChannels = 0
        self._frameArray = None
        
        self._cameraPitchRange = 47.64/180*np.pi
        self._cameraYawRange = 60.97/180*np.pi
        self._cameraProxy.setActiveCamera(self._cameraId)
             
    def updateFrame(self, client_name="video_client"):
        """
        get a new image from the specified camera and save it in self._frame.
        """

        """
        if self._cameraProxy.getActiveCamera() == self._cameraId:
            print("current camera has been actived.")
        else:
            self._cameraProxy.setActiveCamera(self._cameraId)
        """
        self._videoClient = self._cameraProxy.subscribe(client_name, self._resolution, self._colorSpace, self._fps)
        frame = self._cameraProxy.getImageRemote(self._videoClient)
        self._cameraProxy.unsubscribe(self._videoClient)
      
        self._frameWidth = frame[0]
        self._frameHeight = frame[1]
        self._frameChannels = frame[2]
        self._frameArray = np.frombuffer(frame[6], dtype=np.uint8).reshape([frame[1],frame[0],frame[2]])
         
    def getFrameArray(self):
		"""
		return current frame
		"""
		if self._frameArray is None:
			return np.array([])
		return self._frameArray		
		    
    def showFrame(self, timeMs=1000):
        """
        show current frame image.
        """

        if self._frameArray is None:
            print("please get an image from Nao with the method updateFrame()")
        else:
			cv2.imshow("current frame", self._frameArray)
			cv2.waitKey(timeMs)			
    
    def printFrameData(self):
        """
        print current frame data.
        """
        print("frame height = ", self._frameHeight)
        print("frame width = ", self._frameWidth)
        print("frame channels = ", self._frameChannels)
        print("frame shape = ", self._frameArray.shape)
          
    def saveFrame(self, framePath):
		"""
		save current frame to specified direction.
		
		Arguments:
		framePath -- image path.
		"""
		
		cv2.imwrite(framePath, self._frameArray)
		print("current frame image has been saved in", framePath)
				  
    def setParam(self, paramName=None, paramValue = None):
        raise NotImplementedError
     
    def setAllParamsToDefault(self):
        raise NotImplementedError
        

class MultiObjectDetection(VisualBasis):

    def __init__(self, IP, classes_name, cameraId=vd.kTopCamera, resolution=vd.kVGA):
        super(MultiObjectDetection, self).__init__(IP, cameraId, resolution)
        self._classes_name = classes_name
        self._num_classes = len(classes_name)

        self._common_params = {'image_size': 448, 'num_classes': self._num_classes, 
                'batch_size':1}
        self._net_params = {'cell_size': 7, 'boxes_per_cell':2, 'weight_decay': 0.0005}
        self._net = YoloTinyNet(self._common_params, self._net_params, test=True)
        
    def predict_object(self, image):
        predicts = self._net.inference(image)
        return predicts

    def process_predicts(self, resized_img, predicts, thresh=0.2):
        """
        process the predicts of object detection with one image input.
        
        Args:
            resized_img: resized source image.
            predicts: output of the model.
            thresh: thresh of bounding box confidence.
        Return:
            predicts_dict: {"stick": [[x1, y1, x2, y2, scores1], [...]]}.
        """
        cls_num = self._num_classes
        bbx_per_cell = self._net_params["boxes_per_cell"]
        cell_size = self._net_params["cell_size"]
        img_size = self._common_params["image_size"]
        p_classes = predicts[0, :, :, 0:cls_num]
        C = predicts[0, :, :, cls_num:cls_num+bbx_per_cell] # two bounding boxes in one cell.
        coordinate = predicts[0, :, :, cls_num+bbx_per_cell:] # all bounding boxes position.
        
        p_classes = np.reshape(p_classes, (cell_size, cell_size, 1, cls_num))
        C = np.reshape(C, (cell_size, cell_size, bbx_per_cell, 1))
        
        P = C * p_classes # confidencefor all classes of all bounding boxes (cell_size, cell_size, bounding_box_num, class_num) = (7, 7, 2, 1).
        
        predicts_dict = {}
        for i in range(cell_size):
            for j in range(cell_size):
                temp_data = np.zeros_like(P, np.float32)
                temp_data[i, j, :, :] = P[i, j, :, :]
                position = np.argmax(temp_data) # refer to the class num (with maximum confidence) for every bounding box.
                index = np.unravel_index(position, P.shape)
                
                if P[index] > thresh:
                    class_num = index[-1]
                    coordinate = np.reshape(coordinate, (cell_size, cell_size, bbx_per_cell, 4)) # (cell_size, cell_size, bbox_num_per_cell, coordinate)[xmin, ymin, xmax, ymax]
                    max_coordinate = coordinate[index[0], index[1], index[2], :]
                    
                    xcenter = max_coordinate[0]
                    ycenter = max_coordinate[1]
                    w = max_coordinate[2]
                    h = max_coordinate[3]
                    
                    xcenter = (index[1] + xcenter) * (1.0*img_size /cell_size)
                    ycenter = (index[0] + ycenter) * (1.0*img_size /cell_size)
                    
                    w = w * img_size 
                    h = h * img_size 
                    xmin = 0 if (xcenter - w/2.0 < 0) else (xcenter - w/2.0)
                    ymin = 0 if (xcenter - w/2.0 < 0) else (ycenter - h/2.0)
                    xmax = resized_img.shape[0] if (xmin + w) > resized_img.shape[0] else (xmin + w)
                    ymax = resized_img.shape[1] if (ymin + h) > resized_img.shape[1] else (ymin + h)
                    
                    class_name = self._classes_name[class_num]
                    predicts_dict.setdefault(class_name, [])
                    predicts_dict[class_name].append([int(xmin), int(ymin), int(xmax), int(ymax), P[index]])
                    
        return predicts_dict
    
    def non_max_suppress(self, predicts_dict, threshold=0.5):
        """
        implement non-maximum supression on predict bounding boxes.
        Args:
            predicts_dict: {"stick": [[x1, y1, x2, y2, scores1], [...]]}.
            threshhold: iou threshold
        Return:
            predicts_dict processed by non-maximum suppression
        """
        for object_name, bbox in predicts_dict.items():
            bbox_array = np.array(bbox, dtype=np.float)
            x1, y1, x2, y2, scores = bbox_array[:,0], bbox_array[:,1], bbox_array[:,2], bbox_array[:,3], bbox_array[:,4]
            areas = (x2-x1+1) * (y2-y1+1)
            order = scores.argsort()[::-1]
            keep = []
            while order.size > 0:
                i = order[0]
                keep.append(i)
                xx1 = np.maximum(x1[i], x1[order[1:]])
                yy1 = np.maximum(y1[i], y1[order[1:]])
                xx2 = np.minimum(x2[i], x2[order[1:]])
                yy2 = np.minimum(y2[i], y2[order[1:]])
                inter = np.maximum(0.0, xx2-xx1+1) * np.maximum(0.0, yy2-yy1+1)
                iou = inter/(areas[i]+areas[order[1:]]-inter)
                indexs = np.where(iou<=threshold)[0]
                order = order[indexs+1]
            bbox = bbox_array[keep]
            predicts_dict[object_name] = bbox.tolist()
            predicts_dict = predicts_dict
        return predicts_dict

    def plot_result(self, src_img, predicts_dict, save_name = None):
        """
        plot bounding boxes on source image.
        Args:
            src_img: source image
            predicts_dict: {"stick": [[x1, y1, x2, y2, scores1], [...]]}.
        """
        height_ratio = 1.0*src_img.shape[0]/self._common_params["image_size"]
        width_ratio = 1.0*src_img.shape[1]/self._common_params["image_size"]
        for object_name, bbox in predicts_dict.items():
            for box in bbox:
                xmin, ymin, xmax, ymax, score = box
                src_xmin = xmin * width_ratio
                src_ymin = ymin * height_ratio
                src_xmax = xmax * width_ratio
                src_ymax = ymax * height_ratio
                score = float("%.3f" %score)

                cv2.rectangle(src_img, (int(src_xmin), int(src_ymin)), (int(src_xmax), int(src_ymax)), (0, 0, 255))
                cv2.putText(src_img, object_name + str(score), (int(src_xmin), int(src_ymin)), 1, 2, (0, 0, 255))

        cv2.imshow("result", src_img)
        if save_name is not None:
            cv2.imwrite(save_name, src_img)

    def object_track(self, predicts_dict, object_name="cup"):
        """track the specified object with maximum confidence.
        Args:
            object_name: object name.
        """
        if self._motionProxy.getStiffnesses("Head")<1.0:
            self._motionProxy.setStiffnesses("Head", 1.0)

        if self._motionProxy.getStiffnesses("LArm")<1.0:
            self._motionProxy.setStiffnesses("LArm", 1.0)
        img_size = self._common_params["image_size"]
        img_center_x = img_size/2
        img_center_y = img_size/2

        if predicts_dict.has_key(object_name):
            predict_coords = predicts_dict[object_name]
            predict_coords.sort(key=lambda coord:coord[-1], reverse=True)
            predict_coord = predict_coords[0]
            xmin, ymin, xmax, ymax, _ = predict_coord
            center_x = (xmin+xmax)/2
            center_y = (ymin+ymax)/2
            
            angle_yaw = (center_x-img_center_x)/(img_size)*self._cameraYawRange
            angle_pitch = (center_y-img_center_y)/(img_size)*self._cameraPitchRange
            self._motionProxy.angleInterpolation(["HeadPitch","HeadYaw"], [0.8*angle_pitch, -0.8*angle_yaw], 0.5, False)
            head_pitch, head_yaw = self._motionProxy.getAngles("Head", False)
            arm_angle = [head_yaw-7/180*np.pi, head_pitch, -1.15, -0.035, -1.54, 0.01]
            self._motionProxy.setAngles("LArm", arm_angle, 0.2)
            self._motionProxy.openHand("LHand")