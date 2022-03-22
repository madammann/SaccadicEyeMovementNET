import os
from pycocotools.coco import COCO
import requests
import tensorflow_datasets as tfds
import tensorflow as tf
import numpy as np

def coco_dataset_download(class_name,images_count,annotations_path, image_directory, prefix):
    
    parent_dir =prefix+'coco_dataset_subclass'
    paths = os.path.join(parent_dir, image_directory)
    path_with_class = os.path.join(paths, class_name)

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
        

def manual_download_subgroup(names, dir_path,annotations_path, prefix):
    
    class_name=names  #class name example 
    images_count= 10       #count of images  

    #call download function 
    coco_dataset_download(class_name,images_count,annotations_path, dir_path, prefix)
    
   

def download(classes, prefix):
    if not os.path.exists(prefix+'coco_dataset_subclass'):
        for i in classes:
            manual_download_subgroup(i,"train",'./annotations/instances_train2014.json', prefix)

        # for creating the test data
        for i in classes:
            manual_download_subgroup(i,"test",'./annotations/instances_val2014.json', prefix)
    else:
        print("Loading in existing data!")
        
