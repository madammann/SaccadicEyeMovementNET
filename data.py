import tensorflow_datasets as tfds
import numpy as np
import tensorflow as tf

def load_manual_alternative():
    pass

def load_coco(path='./TensorFlowDataSets/'):
    coco_data_raw = None
    try:
        coco_data_raw = tfds.load('coco',download=False,data_dir=path)
    except Excetion as e:
        print(e)
        coco_data_raw = load_manual_alternative()

def preprocess_coco():
    pass
