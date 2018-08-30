#coding: utf-8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import math
import random
import cv2
import numpy as np
from Queue import Queue 
from threading import Thread

from yolo.dataset.dataset import DataSet 

class TextDataSet(DataSet):
  """TextDataSet
  process text input file dataset 
  text file format:
    image_path xmin1 ymin1 xmax1 ymax1 class1 xmin2 ymin2 xmax2 ymax2 class2
  """

  def __init__(self, common_params, dataset_params):
    """
    Args:
      common_params: A dict
      dataset_params: A dict
    """
    #process params
    self.data_path = str(dataset_params['path'])
    self.width = int(common_params['image_size'])
    self.height = int(common_params['image_size'])
    self.batch_size = int(common_params['batch_size'])
    self.thread_num = int(dataset_params['thread_num'])
    self.max_objects = int(common_params['max_objects_per_image'])

    #record and image_label queue
    self.record_queue = Queue(maxsize=10000)
    self.image_label_queue = Queue(maxsize=512)

    self.record_list = []  

    # filling the record_list
    input_file = open(self.data_path, 'r')
	
	##读取txt标签数据并保存到record_list列表中
	##ss格式：[image_path,xmin1,ymin1,xmax1,ymax1,class1,xmin2,ymin2,xmax2,ymax2,class2]
    for line in input_file:
      line = line.strip()
      ss = line.split(' ')
      ss[1:] = [float(num) for num in ss[1:]]
      self.record_list.append(ss)

    self.record_point = 0
    self.record_number = len(self.record_list)

    self.num_batch_per_epoch = int(self.record_number / self.batch_size)
	
	##一个线程用于将数据从record_list中不断读取出数据到record_queue队列中
    t_record_producer = Thread(target=self.record_producer)
    t_record_producer.daemon = True
    t_record_producer.start()
	
	##用多个线程根据record_queue中记录的数据信息，读取图片和标签，并加入到image_label_queue队列中
    for i in range(self.thread_num):
      t = Thread(target=self.record_customer)
      t.daemon = True
      t.start() 

  def record_producer(self):
    """record_queue's processor
    """
    ## 持续将record_list中的数据逐行送到队列中，
    ## 每遍历完一遍数据，将数据重新打乱后重复读取操作，
    ## 当队列满时，等待队列中有数据读出后继续执行（put(item, block=True)）
    while True:
      if self.record_point % self.record_number == 0:
        random.shuffle(self.record_list)
        self.record_point = 0
      self.record_queue.put(self.record_list[self.record_point])
      self.record_point += 1

  def record_process(self, record):
    """record process 
    Args: record 
    Returns:
      image: 3-D ndarray
      labels: 2-D list [self.max_objects, 5] (xcenter, ycenter, w, h, class_num)
      object_num:  total object number  int 
    """
    ##根据record中的数据信息（图片路径，目标矩形框）输出对应图像和标签（目标矩形框位置及类别）
    
    image = cv2.imread(record[0])
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    h = image.shape[0]
    w = image.shape[1]
    
	##读入的图片统一resize成设定的尺寸
    width_rate = self.width * 1.0 / w 
    height_rate = self.height * 1.0 / h 

    image = cv2.resize(image, (self.height, self.width))

	##将box信息转换成[xcenter, ycenter, box_w, box_h, class_num]表示方法
    labels = [[0, 0, 0, 0, 0]] * self.max_objects
    i = 1
    object_num = 0
    while i < len(record):
      xmin = record[i]
      ymin = record[i + 1]
      xmax = record[i + 2]
      ymax = record[i + 3]
      class_num = record[i + 4]

      xcenter = (xmin + xmax) * 1.0 / 2 * width_rate
      ycenter = (ymin + ymax) * 1.0 / 2 * height_rate

      box_w = (xmax - xmin) * width_rate
      box_h = (ymax - ymin) * height_rate

      labels[object_num] = [xcenter, ycenter, box_w, box_h, class_num]
      object_num += 1
      i += 5
      if object_num >= self.max_objects:
        break
    return [image, labels, object_num]

  def record_customer(self):
    """record queue's customer 
    """
    ##持续从record_queue队列中逐条读出数据
    ##将根据数据信息读取对应的图片和标签信息
    ##将图片和对应的标签逐条送到image_label_queue队列中
    while True:
      item = self.record_queue.get()
      out = self.record_process(item)
      self.image_label_queue.put(out)

  def batch(self):
    """get batch
    Returns:
      images: 4-D ndarray [batch_size, height, width, 3]
      labels: 3-D ndarray [batch_size, max_objects, 5]
      objects_num: 1-D ndarray [batch_size]
    """
    images = []
    labels = []
    objects_num = []
    for i in range(self.batch_size):
      image, label, object_num = self.image_label_queue.get()
      images.append(image)
      labels.append(label)
      objects_num.append(object_num)
    images = np.asarray(images, dtype=np.float32)
    images = images/255 * 2 - 1 # 归一化[-1,1]
    labels = np.asarray(labels, dtype=np.float32)
    objects_num = np.asarray(objects_num, dtype=np.int32)
    return images, labels, objects_num
