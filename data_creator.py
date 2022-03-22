import os
from coco_dataset import coco_dataset_download as cocod
import tensorflow_datasets as tfds
import tensorflow as tf
import numpy as np



# create directory and sub directory for storing the data
parent_dir ='coco_dataset_subclass' 
os.makedirs(parent_dir)
dir_train = "train"
dir_test = "test"
paths = os.path.join(parent_dir, dir_train)
os.makedirs(paths)
paths = os.path.join(parent_dir, dir_test)
os.makedirs(paths)

# classes for coco_dataset to load
#load_classes = ['airplane', "apple", "banana",'person','cat','dog','elephant']
load_classes = ['airplane', 'elephant', 'apple']
annotations_path_test = './annotations/instances_val2014.json'
annotations_path_train = './annotations/instances_train2014.json'

from pycocotools.coco import COCO
import requests
import os
# instantiate COCO specifying the annotations json path

def coco_dataset_download(class_name,images_count,annotations_path, image_directory):
    
    parent_dir ='coco_dataset_subclass'
    paths = os.path.join(parent_dir, image_directory)
    path_with_class = os.path.join(paths, class_name)
    print(path_with_class)
    print(os.getcwd())
    #os.makedirs(paths)

    coco = COCO(annotations_path)
    # Specify a list of category names of interest
    catIds = coco.getCatIds(catNms=[class_name])
    # Get the corresponding image ids and images using loadImgs
    imgIds = coco.getImgIds(catIds=catIds)
    images = coco.loadImgs(imgIds)
    os.makedirs(path_with_class)
    # Save the images into a local folder
    count=0
    # specified count images for class name
    for im in images:
        img_data = requests.get(im['coco_url']).content
        with open('./'+ path_with_class +'/'+ im['file_name'], 'wb') as handler:
            handler.write(img_data)
        count+=1
        if count>images_count:
            print('finished images download  -->  {}'.format(class_name) )
            
            break
        #print('no.of image:',count)
        
#from coco_dataset import coco_dataset_download as cocod


def manual_download_subgroup(names, dir_path,annotations_path):
    
    class_name=names  #class name example 
    images_count= 2       #count of images  
    
    #path of coco dataset annotations
    #annotations_path='./annotations/instances_train2014.json' 

     
    #call download function 
    coco_dataset_download(class_name,images_count,annotations_path, dir_path)
    
# for creating the train data    
    
for i in load_classes:
    manual_download_subgroup(i,dir_train,annotations_path_train)

# for creating the test data
for i in load_classes:
    manual_download_subgroup(i,dir_test,annotations_path_test)