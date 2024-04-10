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
from tensorflow import keras
from keras import layers, models
from keras.models import Sequential
from keras.preprocessing import image
import cv2

print('')
print('')

#Directories
coco_root = 'CoCoDataSet/'
tensor_root = 'DataSetTensorFlowStructure/'
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

# batch_size = 8
img_height = 640
img_width = 640

# model = models.Sequential([
#     # layers.Rescaling(1./255, input_shape=(img_height, img_width, 3)),
#     layers.Conv2D(8, (3, 3),input_shape = (640,640,3), activation='relu'),
#     layers.MaxPooling2D((2, 2)),
#     layers.Conv2D(16, (3, 3), activation='relu'),
#     layers.MaxPooling2D((2, 2)),
#     layers.Conv2D(32, (3, 3), activation='relu'),
#     layers.Flatten(),
#     layers.Dense(4, activation='relu'),
#     # layers.Dense(len(class_names), activation='sigmoid')
# ])

# model = models.Sequential([
#     # Initial Convolution and Max Pooling layers
#     layers.Conv2D(16, (3, 3), activation='relu', input_shape=(img_height, img_width, 3)),
#     layers.MaxPooling2D(2, 2),
    
#     # Adding more depth with more Convolutional Layers
#     layers.Conv2D(32, (3, 3), activation='relu'),
#     layers.MaxPooling2D(2, 2),
#     layers.Conv2D(64, (3, 3), activation='relu'),
#     layers.MaxPooling2D(2, 2),
    
#     # Additional layers with increased filters
#     layers.Conv2D(128, (3, 3), activation='relu'),
#     layers.MaxPooling2D(2, 2),
#     layers.Conv2D(64, (3, 3), activation='relu'),
#     layers.MaxPooling2D(2, 2),
#     layers.Conv2D(32, (3, 3), activation='relu'),
#     layers.MaxPooling2D(2, 2),
#     layers.Conv2D(16, (3, 3), activation='relu'),
#     layers.MaxPooling2D(2, 2),
#     layers.Flatten(),

#     # Dense layers for feature interpretation
#     layers.Dense(512, activation='relu'),
#     layers.Dropout(0.4),  # Increased dropout for regularization
#     # Output layer for bounding box prediction - 4 neurons for [x_center, y_center, width, height]
#     # No activation function is used here to allow for unbounded outputs
#     layers.Dense(4, activation=None)  # Consider removing the activation or using 'linear' if working with unbounded coordinates
# ])

model = models.Sequential([
    # Initial Convolution and Max Pooling layers
    layers.Conv2D(16, (3, 3), activation='relu', padding='same', input_shape=(img_height, img_width, 3)),
    layers.MaxPooling2D(2, 2),
    
    # Adding more depth with more Convolutional Layers
    layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
    layers.MaxPooling2D(2, 2),

    layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
    layers.MaxPooling2D(2, 2),

    layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
    layers.MaxPooling2D(2, 2),

    layers.Conv2D(256, (3, 3), activation='relu', padding='same'),
    layers.MaxPooling2D(2, 2),

    layers.Conv2D(512, (3, 3), activation='relu', padding='same'),
    layers.MaxPooling2D(2, 2),
    # Flattening the output to feed into Dense layers
    layers.Flatten(),

    # Dense layers for feature interpretation
    layers.Dense(512, activation='relu'),
    layers.Dropout(0.33),  # Increased dropout for regularization

    # Output layer for bounding box prediction - 4 neurons for [x_center, y_center, width, height]
    layers.Dense(4, activation=None)  # Consider removing the activation or using 'linear' if working with unbounded coordinates
])

numpy_train_images = pp.PillowImageArrayToNumpyArray(train_images_list, True)
numpy_train_bbox_list = pp.BoundingBoxesToNumpyArray(train_bbox_list)

numpy_test_images = pp.PillowImageArrayToNumpyArray(test_images_list, True)
numpy_test_bbox_list = pp.BoundingBoxesToNumpyArray(test_bbox_list)

numpy_valid_images = pp.PillowImageArrayToNumpyArray(valid_images_list, True)
numpy_valid_bbox_list = pp.BoundingBoxesToNumpyArray(valid_bbox_list)

model.compile(optimizer='adam',
              #loss='mean_squared_error' ,
              loss = 'mean_squared_error',
              # loss='binary_crossentropy',
              metrics=['accuracy'])
# Assuming your labels are correctly shaped, you can now fit the model
H = model.fit(numpy_train_images, numpy_train_bbox_list, epochs = 256, validation_data=(numpy_valid_images, numpy_valid_bbox_list))

validation_loss = model.evaluate(numpy_valid_images, numpy_valid_bbox_list)

S = model.summary()

print(H.history)
print(S)



print(f'validation loss: {validation_loss}')