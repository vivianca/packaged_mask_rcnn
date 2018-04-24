import os
import sys
import random
import math
import numpy as np
import skimage.io
import matplotlib
import matplotlib.pyplot as plt
import rospy

from . import coco
from . import utils
from . import model as modellib
from . import visualize

class_names = ['BG', 'person', 'bicycle', 'car', 'motorcycle', 'airplane',
       'bus', 'train', 'truck', 'boat', 'traffic light',
       'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird',
       'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear',
       'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie',
       'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
       'kite', 'baseball bat', 'baseball glove', 'skateboard',
       'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup',
       'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
       'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
       'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed',
       'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote',
       'keyboard', 'cell phone', 'microwave', 'oven', 'toaster',
       'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors',
       'teddy bear', 'hair drier', 'toothbrush']

class InferenceConfig(coco.CocoConfig):
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1

class EZ():

    def __init__(self):
        self.ROOT_DIR = os.getcwd()
        self.MODEL_DIR = os.path.join(self.ROOT_DIR, "logs")
        self.COCO_MODEL_PATH = os.path.join(self.ROOT_DIR, "mask_rcnn_coco.h5")
        if not os.path.exists(self.COCO_MODEL_PATH):
            utils.download_trained_weights(self.COCO_MODEL_PATH)
        self.config = InferenceConfig()
        self.config.display()
        print('loading model ...')
        self.model = modellib.MaskRCNN(mode="inference", model_dir=self.MODEL_DIR, config=self.config)
        self.model.load_weights(self.COCO_MODEL_PATH, by_name=True)
        self.class_names = class_names
        self.class_colors = visualize.random_colors(len(self.class_names))

    def serve(self, service_name):

        rospy.init_node('serve')
        s = rospy.Service(service_name, self.__class__, req_handler)

        """
        description: this function will start this vision
        system as a ros node, listen to channels and serve
        results by calling self.detect method.

        input:
            service_name: a string name given to the servive

        output:
            none (this function will run forever)
        """

    def req_handler(self, req):

        """
        description: this function is a thin wrapper around
        self.detect and will be used in self.serve

        inputs:
            req: the request object

        outputs:
            res: response to the request
        """

    def detect(self, image_input, merge_image=True):
        sample_image = image_input
        if type(image_input) is str:
            sample_image = skimage.io.imread(image_input)
        if len(sample_image.shape) != 3 or sample_image.shape[2] != 3:
            print('Error: image shoud have dimension (W, H, 3)')
            return None
        results = self.model.detect([sample_image], verbose=1)
        r = results[0]
        output_image = None
        if merge_image:
            output_image = visualize.get_displayed_instances(sample_image, r['rois'],
                            r['masks'], r['class_ids'], class_names, r['scores'], class_colors=self.class_colors)
        info = {
            'masks': r['masks'],
            'class_ids': r['class_ids'],
            'scores': r['scores'],
            'rois': r['rois']
        }
        return output_image, info
