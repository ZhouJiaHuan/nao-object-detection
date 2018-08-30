## ---------------------------------------------------------------------
# author: Meringue
# date: 1/15/2018
# description: common objects detection and tracking for NAO robot with tiny YOLO model.
## ---------------------------------------------------------------------
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import cv2
import time
import os
import sys
sys.path.append("/home/meringue/Softwares/pynaoqi-sdk/") #naoqi directory
sys.path.append("./")

from visualTask import *

from naoqi import ALProxy
import vision_definitions as vd

IP = "192.168.1.156"

#for 3-class detection
classes_name = ["stick", "cup", "pen"]
modelFile = "/home/meringue/Documents/nao-object-detection/yoloNet/models/train/model.ckpt-100000"

#for 20-class detection
#classes_name =  ["aeroplane", "bicycle", "bird", "boat", "bottle", \
#				 "bus", "car", "cat", "chair", "cow", "diningtable", \
#				 "dog", "horse", "motorbike", "person", "pottedplant", \
#				 "sheep", "sofa", "train","tvmonitor"]
#modelFile = "/home/meringue/Documents/nao-object-detection/yoloNet/models/pretrain/yolo_tiny.ckpt"

track_object = "stick"
print("object detection and tracking...")

multiObjectDetect = MultiObjectDetection(IP, classes_name, cameraId=0, resolution=vd.kVGA)
image = tf.placeholder(tf.float32, (1, 448, 448, 3))
object_predicts = multiObjectDetect.predict_object(image)

sess = tf.Session()
saver = tf.train.Saver(multiObjectDetect._net.trainable_collection)
saver.restore(sess, modelFile)

index = 0
while 1:
	#update and preprocess one frame image.
	multiObjectDetect.updateFrame()
	frame = multiObjectDetect.getFrameArray()
	resized_img = cv2.resize(frame, (448, 448))
	height_ratio = frame.shape[0]/448
	width_ratio = frame.shape[1]/448
	np_img = cv2.cvtColor(resized_img, cv2.COLOR_BGR2RGB)
	np_img = np_img.astype(np.float32)
	np_img = np_img / 255.0 * 2 - 1
	np_img = np.reshape(np_img, (1, 448, 448, 3))

	#detection and tracking process
	np_predict = sess.run(object_predicts, feed_dict={image: np_img})
	predicts_dict = multiObjectDetect.process_predicts(resized_img, np_predict)
	predicts_dict = multiObjectDetect.non_max_suppress(predicts_dict)
	print ("predict dict = ", predicts_dict)
	multiObjectDetect.object_track(predicts_dict, track_object)

	#show results
	save_name = None
	#save_name = "result" + str(index) + ".jpg"
	index += 1
	multiObjectDetect.plot_result(frame, predicts_dict, save_name)	
	if cv2.waitKey(10) & 0xFF==ord("q"):
		break
sess.close()