import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import os
import datetime

def draw_bounding_boxes(i, predict_confidence, image, true_boxes, predict_boxes, epoch, save_path = None):
    """
    Draws bounding boxes on image.
    Args:
    image (PIL Image): PIL Image object.
    true_boxes (list): True bounding box [x_min, y_min, width, height].
    predict_boxes (list): Predicted bounding box [x_min, y_min, width, height].
    save_path (str): Directory to save the image.
    """

    # Ensure the save directory exists
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    # Convert PIL Image to numpy array
    np_image = np.array(image)
    fig, ax = plt.subplots(1)
    ax.imshow(np_image)

    # Add predicted bounding box
    # Draw predicted bounding box
    pred_label = f'Prediction: {predict_confidence * 100:.2f}% confidence'
    pred_rect = patches.Rectangle((predict_boxes[0], predict_boxes[1]), predict_boxes[2], predict_boxes[3], linewidth=1, edgecolor='g', facecolor='none', label=pred_label)
    ax.add_patch(pred_rect)
    #rectPredict = patches.Rectangle((predict_boxes[0], predict_boxes[1]), predict_boxes[2], predict_boxes[3], linewidth=1, edgecolor='g', facecolor='none')
    #ax.add_patch(rectPredict)

    # Add true bounding box
    rectTrue = patches.Rectangle((true_boxes[0], true_boxes[1]), true_boxes[2], true_boxes[3], linewidth=1, edgecolor='r', facecolor='none', label='True Box')
    ax.add_patch(rectTrue)

    # Add legend
    ax.legend(loc='upper right')

    # Save the plot to a file
    output_filename = f"{i}_epoch_{epoch}.png"
    output_path = os.path.join(save_path, output_filename)
    plt.axis('off')
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()  # Close the figure to free memory