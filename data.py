import os
import requests

import tensorflow_datasets as tfds
import tensorflow as tf

from pycocotools.coco import COCO
from tqdm import tqdm

from multiprocessing.pool import ThreadPool
import time

def download_image(img,terminate=False):
    try:
        img_data = requests.get(img['coco_url']).content
        with open(img['path_with_class'] +'/'+ img['file_name'], 'wb') as handler:
            handler.write(img_data)
        time.sleep(2)
        return True
    except Exception as e:
        if os.path.exists(img['path_with_class'] +'/'+ img['file_name']):
            return True
        else:
            if terminate == True:
                return False
            else:
                time.sleep(30)
                download_image(img,terminate=True)

def coco_dataset_download(coco, class_name, image_directory, prefix,multiprocessing=True):
    parent_dir =prefix+'coco_dataset_subclass'
    paths = os.path.join(parent_dir, image_directory)
    path_with_class = os.path.join(paths, class_name)

    '''Specify a list of category names of interest'''
    catIds = coco.getCatIds(catNms=[class_name])
    '''Get the corresponding image ids and images using loadImgs'''
    imgIds = coco.getImgIds(catIds=catIds)
    images = coco.loadImgs(imgIds)
    '''Create folder for class'''
    os.makedirs(path_with_class)
    for image in images:
        image['path_with_class'] = path_with_class

    if multiprocessing:
        try:
            thread_pool = ThreadPool(20).imap_unordered(download_image, [images[i] for i in range(len(images))])
            for thread in tqdm(thread_pool, desc='Downloading '+str(class_name)+' images.'):
                res = thread
        except Exception as e:
            print(e)
            print('Executing again with multiprocessing off.')
            coco_dataset_download(coco, class_name, image_directory, prefix,multiprocessing=False)
    else:
        for img in tqdm(images,desc='Downloading '+str(class_name)+' images.'):
            download_image(img)

def download(classes, prefix,multiprocessing=True):
    if not os.path.exists(prefix+'coco_dataset_subclass'):
        coco_train = COCO('./annotations/instances_train2014.json')
        coco_test = COCO('./annotations/instances_val2014.json')
        for class_name in tqdm(classes,desc='Downloading train and test images.'):
            coco_dataset_download(coco_train,class_name,'train',prefix,multiprocessing=multiprocessing)
            coco_dataset_download(coco_test,class_name,'test',prefix,multiprocessing=multiprocessing)
            print('Finished downloading Images for class: '+str(class_name))
    else:
        print("Loading in existing data.")

def load_manual_alternative(prefix):
    builder = tfds.ImageFolder(prefix+'coco_dataset_subclass/')
    dataset = builder.as_dataset(as_supervised=True)
    return dataset

def preprocess_data(dataset, batchsize, numOfClasses):

    coco = coco.map(lambda img, target: (tf.image.resize(img, [128,128],
                                         method = tf.image.ResizeMethod.BILINEAR,
                                         preserve_aspect_ratio=False),target))
    #convert data from uint8 to float32
    dataset = dataset.map(lambda img, target: (tf.cast(img, tf.float32), target))

    #input normalization, just bringing image values from range [0, 255] to [0, 1]
    dataset = dataset.map(lambda img, target: (tf.math.l2_normalize(img),target))

    #create one-hot targets
    dataset = dataset.map(lambda img, target: (img, tf.one_hot(target, depth=numOfClasses)))
    #cache this progress in memory, as there is no need to redo it; it is deterministic after all
    #dataset = dataset.cache()
    #shuffle, batch, prefetch
    dataset = dataset.shuffle(100)
    dataset = dataset.batch(batchsize)
    dataset = dataset.prefetch(2)
    #return preprocessed dataset
    return dataset
