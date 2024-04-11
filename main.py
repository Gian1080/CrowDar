import preprocessing as pp
import postprocessing as post
import directory as dir
import imaging as im
import json
from keras import layers, models
import datetime

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

#convert bounding box to numpy arrays
numpy_train_images = pp.PillowImageArrayToNumpyArray(train_images_list, False)
numpy_train_bbox_list = pp.BoundingBoxesToNumpyArray(train_bbox_list)

numpy_test_images = pp.PillowImageArrayToNumpyArray(test_images_list, False)
numpy_test_bbox_list = pp.BoundingBoxesToNumpyArray(test_bbox_list)

numpy_valid_images = pp.PillowImageArrayToNumpyArray(valid_images_list, False)
numpy_valid_bbox_list = pp.BoundingBoxesToNumpyArray(valid_bbox_list)

img_height = 640
img_width = 640
batch_size = 16
epoch_count = 13

bigModel = models.Sequential([

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


bigModel.compile(optimizer='adam',
              loss = 'mean_absolute_error',
              metrics=['accuracy'])

for image in numpy_test_images:
    pp.AppendImageToNumpyArray(numpy_train_images, image)

for bbox in numpy_test_bbox_list:
    pp.AppenBoundingBoxToNumpyArray(numpy_train_bbox_list, bbox)

#fill with prime numbers
crow_epochs = [1, 3, 5]
# Get the current date and time
now = datetime.datetime.now()
date_time = now.strftime("%Y-%m-%d_%H-%M-%S")

for epoch in crow_epochs:    
    # create a unique directory for each epoch
    unique_dir_name = dir.create_unique_result_directory(epoch, date_time)
    # fit/train the model
    H = bigModel.fit(numpy_train_images, numpy_train_bbox_list, epochs = epoch, batch_size=batch_size, validation_data=(numpy_valid_images, numpy_valid_bbox_list))
    validation_loss = bigModel.evaluate(numpy_valid_images, numpy_valid_bbox_list)
    S = bigModel.summary()
    print(H.history)
    print(S)

    # Assuming `numpy_valid_images` is your validation set images and the model is named `smallModel`
    predictions = bigModel.predict(numpy_valid_images)
    average = 0
    counter = 0
    for i in range(len(numpy_valid_images)):
        iou = post.CalculateIoUNEW(predictions[i], numpy_valid_bbox_list[i])
        counter += 1
        average += iou
        im.draw_bounding_boxes(i, iou, numpy_valid_images[i], numpy_valid_bbox_list[i], predictions[i], epoch, unique_dir_name)
        
    print(f'average iou: {average/counter} @ epoch {epoch}')