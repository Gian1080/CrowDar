import cocoHelper
import tfHelper
import pathlib
import os
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import tkinter as tk
import tensorflow as tf
import os
import cv2
from tkinter import filedialog
from pycocotools.coco import COCO
from matplotlib import pyplot as plt
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
#from keras.preprocessing.image import ImageDataGenerator



model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 3)),
    MaxPooling2D(2, 2),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Flatten(),
    Dense(512, activation='relu'),
    Dropout(0.5),
    Dense(1, activation='sigmoid')  # Gebruik 'sigmoid' voor twee klassen; vervang door 'softmax' voor meerdere klassen
])

# train_datagen = ImageDataGenerator(rescale=1./255, rotation_range=40, width_shift_range=0.2,
#                                    height_shift_range=0.2, shear_range=0.2, zoom_range=0.2,
#                                    horizontal_flip=True, fill_mode='nearest')

# model.compile(optimizer='adam',
#               loss='binary_crossentropy',  # Gebruik 'categorical_crossentropy' voor meerdere klassen
#               metrics=['accuracy'])
# Initialiseer Tkinter root widget
root = tk.Tk()
root.withdraw()  # We willen niet de volledige GUI, dus sluiten we het hoofdvenster

# Open een dialoogvenster om de annotaties.json te selecteren
annotations_path = filedialog.askopenfilename(
    title="Selecteer het COCO annotaties bestand (_annotations.coco.json)"
)

# Open een dialoogvenster om de afbeeldingen directory te selecteren
images_dir = filedialog.askdirectory(
    title="Selecteer de map met COCO afbeeldingen"
)

# Controleer of de paden geldig zijn
if not os.path.exists(annotations_path) or not os.path.exists(images_dir):
    raise ValueError("Een van de geselecteerde paden bestaat niet.")

# Laad de COCO annotaties
coco = COCO(annotations_path)

# Krijg alle afbeelding IDs
image_ids = coco.getImgIds()

# Loop door een subset van afbeeldingen (vervang len(image_ids) met een kleiner getal om te testen)
for i in range(len(image_ids)):
    # Verkrijg afbeelding metadata
    img_info = coco.loadImgs(image_ids[i])[0]
    
    # Pad naar de afbeelding
    img_path = os.path.join(images_dir, img_info['file_name'])
    
    # Laad de afbeelding
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Converteer BGR naar RGB
    
    # Toon de afbeelding
    plt.imshow(img)
    plt.axis('off')
    plt.show()
    
    # Stop na het tonen van één afbeelding voor dit voorbeeld
    #break
