import tensorflow_datasets as tfds
import tensorflow as tf

def load_manual_alternative(prefix):
    builder = tfds.ImageFolder(prefix+'coco_dataset_subclass/')
    print(builder.info)
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



