import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

def draw_bounding_boxes(image, true_boxes, predict_boxes):
    """
    Draws bounding boxes on image.
    image: PIL Image object.
    boxes: Array of boxes in format [x_min, y_min, x_max, y_max] or [x_center, y_center, width, height].
    """
    # Convert PIL Image to numpy array
    np_image = np.array(image)
    fig, ax = plt.subplots(1)
    ax.imshow(np_image)

    # Convert from [x_center, y_center, width, height] if necessary
    rectPredict = patches.Rectangle((predict_boxes[0], predict_boxes[1]), predict_boxes[2], predict_boxes[3], linewidth=1, edgecolor='g', facecolor='none')
    ax.add_patch(rectPredict)
    rectTrue = patches.Rectangle((true_boxes[0], true_boxes[1]), true_boxes[2], true_boxes[3], linewidth=1, edgecolor='r', facecolor='none')
    ax.add_patch(rectTrue)

    plt.show()
    
def calculate_iou(box1, box2):
    """
    Calculate Intersection over Union (IoU) of two bounding boxes.

    Arguments:
    box1: Tuple (x1, y1, x2, y2) representing coordinates of the first bounding box.
    box2: Tuple (x1, y1, x2, y2) representing coordinates of the second bounding box.

    Returns:
    float: IoU value.
    """
    # Extract coordinates of the bounding boxes
    x1_1, y1_1, x2_1, y2_1 = box1
    x1_2, y1_2, x2_2, y2_2 = box2

    # Calculate the coordinates of the intersection rectangle
    x_left = max(x1_1, x1_2)
    y_top = max(y1_1, y1_2)
    x_right = min(x2_1, x2_2)
    y_bottom = min(y2_1, y2_2)

    # If the boxes don't intersect, return 0
    if x_right < x_left or y_bottom < y_top:
        return 0.0

    # Calculate intersection area
    intersection_area = (x_right - x_left) * (y_bottom - y_top)

    # Calculate areas of the bounding boxes
    area_box1 = (x2_1 - x1_1) * (y2_1 - y1_1)
    area_box2 = (x2_2 - x1_2) * (y2_2 - y1_2)

    # Calculate union area
    union_area = area_box1 + area_box2 - intersection_area

    # Calculate IoU
    iou = intersection_area / union_area

    return iou

def calculate_overlap_boxes(true_box, predicted_box):
    true_pixels = []
    startX = int(true_box[0])
    endX = int(true_box[1])
    
    startY = int(true_box[2])
    endY = int(true_box[3])
    for x in range(startX, endX):
        for y in range(startY, endY):
            true_pixels.append((x, y))