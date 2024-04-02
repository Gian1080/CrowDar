import tkinter as tk
from tkinter import filedialog
import os
from pycocotools.coco import COCO

def select_directory(title="Select Folder"):
    """
    Open een dialoogvenster voor de gebruiker om een map te selecteren.

    Args:
    - title: De titel van het dialoogvenster.

    Returns:
    - Het geselecteerde directorypad.
    """
    root = tk.Tk()
    root.withdraw()  # We willen niet de volledige GUI, dus sluiten we het hoofdvenster
    folder_selected = filedialog.askdirectory(title=title)
    return folder_selected

def load_coco_data(base_dir, dataset_type):
    """
    Laad COCO data van een specifiek type (train, test, valid).

    Args:
    - base_dir: De basisdirectory waar de COCO dataset is opgeslagen.
    - dataset_type: Een van 'train', 'test', of 'valid'.

    Returns:
    - coco: De COCO object instantie.
    - image_dir: Het pad naar de afbeeldingen directory.
    - annotation_file: Het pad naar het annotatiebestand.
    """
    annotation_file = os.path.join(base_dir, dataset_type, "_annotations.coco.json")
    image_dir = os.path.join(base_dir, dataset_type)

    coco = COCO(annotation_file)

    return coco, image_dir, annotation_file