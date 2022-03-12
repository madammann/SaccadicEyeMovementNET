import os
import tensorflow_datasets as tfds

def prepare_coco_dataset(mode='automatic', path='./TensorFlowDataSets/'):
    coco_data_raw = None
    '''Implement check for storage capacity at location path!'''
    if mode == 'automatic':
        coco_data_raw = tfds.load('coco',data_dir=path)
    else:
        pass
