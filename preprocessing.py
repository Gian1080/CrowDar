import os
import numpy as np
import matplotlib as mpl
import json
from PIL import Image
from tkinter import filedialog


def LoadImageFromPath(path):
    return Image.open(path)

def LoadImages(paths):
    return {id : LoadImageFromPath(paths[id]) for id in paths}


def GetIDFromImage(file_name, json_data):
    image_list = json_data["images"]
    for image in image_list:
        if image["file_name"] == file_name:
            return image["id"]
        
def GetBoundingBoxFromID(ID, json_data):
    pass

def GetCleanedTrainPaths(paths, prefix):
    return [path.replace(f"CocoDataSet/{prefix}\\", "") for path in paths]

def CountSingleCrows(json_data):
    number_of_images = len(json_data['images']) 
    count = 0
    for i in range(number_of_images+1):   
        number_of_crows = 0 
        for datapoint in json_data['annotations']:
            if datapoint['image_id'] == i:
                number_of_crows+=1
        if number_of_crows ==1:
            count+=1
                

def GetSingleCrowsIDs(json_data):
    number_of_images = len(json_data['images']) 
    IDs = []
    for id in range(number_of_images+1):   
        number_of_crows = 0 
        for datapoint in json_data['annotations']:
            if datapoint['image_id'] == id:
                number_of_crows += 1
        if number_of_crows == 1:
            IDs.append(id)
    print(len(IDs))
    return IDs

def GetListOfPaths(IDs, json_data, prefix):
    paths = {}
    for image in json_data['images']:
        for ID in IDs:
            if image['id'] == ID:
                paths[ID] = (f"{prefix}{image['file_name']}")
                break
    return paths

def GetBoundingBoxByID(ID, json_data):
    for annotation in json_data['annotations']:
        if annotation['id'] == ID:
            return annotation['bbox']
        
def GetBoundingBoxesByIDs(IDs, json_data):
    return {id:GetBoundingBoxByID(id, json_data) for id in IDs}      