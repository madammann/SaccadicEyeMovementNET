import tensorflow_datasets as tfds
import numpy as np
import tensorflow as tf
import os

def load_manual_alternative():
    print(os.getcwd())
    builder = tfds.ImageFolder('./coco_dataset_subclass/')
    print(os.getcwd())
    print(builder.info)
    dataset = builder.as_dataset(shuffle_files=True, as_supervised=True)
    
    train_ds, test_ds = dataset["train"], dataset["test"]
    
    return train_ds, test_ds

# def load_coco(path='./TensorFlowDataSets/'):
#     coco_data_raw = None
#     try:
#         coco_data_raw = tfds.load('coco',download=False,data_dir=path)
#     except Excetion as e:
#         print(e)
#         coco_data_raw = load_manual_alternative()

# def preprocess_coco():
#     pass

    
def prepare_coco_data(coco):
    #flatten the images into vectors
    coco = coco.map(lambda img, target: (tf.image.resize(img, [128,128],
                                         method = tf.image.ResizeMethod.BILINEAR, 
                                         preserve_aspect_ratio=False),      target))
  
    #convert data from uint8 to float32
    coco = coco.map(lambda img, target: (tf.cast(img, tf.float32), target))
    #sloppy input normalization, just bringing image values from range [0, 255] to [-1, 1]
    #coco = coco.map(lambda img, target: ((img/128.)-1., target))

    coco = coco.map(lambda img, target: (tf.math.l2_normalize(img),target))


    #create one-hot targets
    coco = coco.map(lambda img, target: (img, tf.one_hot(target, depth=3)))
    #cache this progress in memory, as there is no need to redo it; it is deterministic after all
    coco = coco.cache()
    #shuffle, batch, prefetch
    coco = coco.shuffle(1000)
    coco = coco.batch(4)
    coco= coco.prefetch(20)
    #return preprocessed dataset
    return coco

# load the train and test
train_ds, test_ds = load_manual_alternative()

# preprocess the train and test

train_dataset= train_ds.apply(prepare_coco_data)
test_dataset = test_ds.apply(prepare_coco_data)

print(train_dataset)


