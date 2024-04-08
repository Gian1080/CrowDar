import preprocessing as pp
from pycocotools.coco import COCO
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import numpy as np
import json
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

import tensorflow as tf
import object_detection
from object_detection.utils import config_util
from object_detection.builders import model_builder
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as viz_utils
from object_detection.utils import ops as utils_ops


print('')
print('')
#Directories
coco_root = 'CoCoDataSet/'
annotation_dir = '_annotations.coco.json'
train_dir = f'{coco_root}train/'
test_dir = f'{coco_root}test/'
valid_dir = f'{coco_root}valid/'

#inladen JSON annotaties
train_annotation = json.load(open(f'{train_dir}{annotation_dir}'))
test_annotation = json.load(open(f'{test_dir}{annotation_dir}'))
valid_annotation = json.load(open(f'{valid_dir}{annotation_dir}'))

#ID nummers van geschikte foto's
train_ids = pp.GetSingleCrowsIDs(train_annotation)
test_ids = pp.GetSingleCrowsIDs(test_annotation)
valid_ids = pp.GetSingleCrowsIDs(valid_annotation)

#Paden naar afbeeldingen bij geschikte foto's
train_paths = pp.GetListOfPaths(train_ids, train_annotation, train_dir)
test_paths = pp.GetListOfPaths(test_ids, test_annotation, test_dir)
valid_paths = pp.GetListOfPaths(valid_ids, valid_annotation, valid_dir)

#Geschikte foto's van enkele kraaien
train_images = pp.LoadImages(train_paths)
test_images = pp.LoadImages(test_paths)
valid_images = pp.LoadImages(valid_paths)

#bijbehorende bounding boxes
train_bbox = pp.GetBoundingBoxesByIDs(train_ids, train_annotation)
test_bbox = pp.GetBoundingBoxesByIDs(test_ids, test_annotation)
valid_bbox = pp.GetBoundingBoxesByIDs(valid_ids, valid_annotation)

#gesorteerde id, foto's en bounding boxes
train_IDS = train_images.keys()
train_images_list = [train_images[id] for id in train_IDS]
train_bbox_list  = [train_bbox[id] for id in train_IDS]

test_IDS  = test_images.keys()
test_images_list = [test_images[id] for id in test_IDS]
test_bbox_list  = [test_bbox[id] for id in test_IDS]

valid_IDS = valid_images.keys()
valid_images_list = [valid_images[id] for id in valid_IDS]
valid_bbox_list  = [valid_bbox[id] for id in valid_IDS]

