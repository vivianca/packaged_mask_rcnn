import numpy as np

from .config import Config
from . import utils
from .coco import CocoDataset

coco_class_names = ['BG', 'person', 'bicycle', 'car', 'motorcycle', 'airplane',
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

extend_class_names = ['glasses', 'rug', 'pill bottle', 'pen']

class ExtendConfig(Config):

    NAME = 'extend'

    GPU_COUNT = 1
    IMAGES_PER_GPU = 8

    NUM_CLASSES = 1 + len(coco_class_names) + len(extend_class_names)

class ExtendDataset(CocoDataset):

    def load_extend_dataset(self, coco_year="2014", extend_dir='dataset/extend'):
        """
        This function should first call self.load_coco and then use the
        extend_dir to load extended dataset and append to coco dataset
        note: images should be added using self.add_image funtion so that
        it doesn't conflict with parent methods
        """
        pass

    def load_mask(self, image_id):
        """
        This function first need to check which dataset does this
        image_id referring to, either coco dataset or extended dataset
        if it is pointing to coco dataset, call parent method:
            return super(ExtendDataset, self).load_mask(image_id)
        if it is pointing to extended dataset, load mask from the
        extended dataset directory
        """
        pass

    def image_reference(self, image_id):
        """
        This function first need to check which dataset does this
        image_id referring to, either coco dataset or extended dataset
        if it is pointing to coco dataset, call parent method:
            return super(ExtendDataset, self).image_reference(image_id)
        if it is pointing to extended dataset, load the image reference
        from the extended dataset directory
        """
        pass
