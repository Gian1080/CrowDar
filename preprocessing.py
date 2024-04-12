import numpy as np
from PIL import Image

def load_image_from_path(path):
    """Load an image from a given file path."""
    return Image.open(path)

def load_images(paths):
    """Load multiple images given a dictionary of ID to file paths."""
    return {id_: load_image_from_path(paths[id_]) for id_ in paths}

def get_id_from_image(file_name, json_data):
    """Retrieve the ID for an image given its file name from JSON data."""
    for image in json_data["images"]:
        if image["file_name"] == file_name:
            return image["id"]
    return None

def get_bounding_box_by_id(id_, json_data):
    """Retrieve the bounding box for an image ID from JSON data."""
    for annotation in json_data['annotations']:
        if annotation['image_id'] == id_:
            return annotation['bbox']
    return None

def get_bounding_boxes_by_ids(ids, json_data):
    """Retrieve bounding boxes for multiple IDs."""
    return {id_: get_bounding_box_by_id(id_, json_data) for id_ in ids}

def clean_train_paths(paths, prefix):
    """Clean the file paths by removing the prefix for Coco Dataset."""
    return [path.replace(f"CocoDataSet/{prefix}\\", "") for path in paths]

def count_single_crows(json_data):
    """Count images with exactly one crow based on annotations."""
    count = 0
    number_of_images = len(json_data['images'])
    for i in range(number_of_images + 1):
        number_of_crows = sum(1 for dp in json_data['annotations'] if dp['image_id'] == i)
        if number_of_crows == 1:
            count += 1
    return count

def get_single_crows_ids(json_data):
    """Get IDs of images that have exactly one crow."""
    ids = []
    number_of_images = len(json_data['images'])
    for id_ in range(number_of_images + 1):
        number_of_crows = sum(1 for dp in json_data['annotations'] if dp['image_id'] == id_)
        if number_of_crows == 1:
            ids.append(id_)
    return ids

def pillow_image_array_to_numpy_array(list_of_images, normalize=False):
    """Convert a list of PIL Images to a numpy array, with optional normalization."""
    numpy_array = np.array([np.array(image) for image in list_of_images])
    if normalize:
        numpy_array = numpy_array.astype('float32') / 255.0
    return numpy_array

def bounding_boxes_to_numpy_array(bounding_boxes):
    """Convert a list of bounding boxes to a numpy array."""
    return np.array(bounding_boxes)

def append_image_to_numpy_array(numpy_array, image):
    """Append an image to a numpy array of images."""
    return np.append(numpy_array, np.array(image), axis=0)

def append_bounding_box_to_numpy_array(numpy_array, bounding_box):
    """Append a bounding box to a numpy array of bounding boxes."""
    return np.append(numpy_array, bounding_box, axis=0)
