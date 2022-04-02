'''Module imports'''

'''Library imports'''

import cv2 as cv
from matplotlib import pyplot as plt
import sys
import numpy as np
import colorsys
from data import *

def get_pictures(prefix):
    picture_file = None
    with open('./setup/pictures.txt','r') as file:
        picture_file = file.read().split('\n')
    try:
        picture_file = [(line.split(',')[0], line.split(',')[1]) for line in picture_file if line != ""]
        print(line for line in picture_file if line != "")
        return [(tf.convert_to_tensor(cv.imread(prefix+elem[0])),tf.convert_to_tensor(tf.onehot(int(elem[1]), depth=11))) for elem in picture_file]
    except Exception as e:
        print(e)
        raise FileNotFoundError('Unable to read picture file.')

def visualize_posttraining(image, model):
    '''Draws the saccadic eye movement for an example image and classification.'''
    pass

def visualize_midtraining():
    '''Draws the benchmark performance for a training epoch.'''
    pass

def plot_lossaccuracy():
    '''Draws the loss and accuracy of both models over the epochs.'''
    df = pd.read_csv('./eval/loss_accuracy.csv')
    snc_df = df.where(df['Model'] == 'SaccadicNetClassifier').dropna()
    sne_df = df.where(df['Model'] == 'SaccadicNetEye').dropna()
    fig, ax = plt.subplots(1,2,figsize=(15, 4))
    line1, = ax[0].plot(snc_df['Loss'].values, label='SaccadicNetClassifier')
    line2, = ax[0].plot(sne_df['Loss'].values, label='SaccadicNetEye')
    line3, = ax[1].plot(snc_df['Accuracy'].values, label='SaccadicNetClassifier')
    ax[0].set_xlabel("Training epoch")
    ax[0].set_ylabel("Loss")
    ax[1].set_xlabel("Training epoch")
    ax[1].set_ylabel("Accuracy")
    ax[0].legend()
    ax[1].legend()
    plt.show()


def plot_movement():
    '''Plots an analysis of the movement behavior for comparison with the human eye'''
    pass

def webcam_movement_record():
    '''Records movement of a viewer of an image'''
    pass

def compare_movement():
    '''Analyses movement of human and net by comparison'''
    pass

def hsv2rgb(h,s,v):

  '''for changing the hsv values into BGR'''
  return tuple(round(i * 255) for i in colorsys.hsv_to_rgb(h,s,v))


def visualize_image_patches(image, cordinates, patch_count=10):
    ''' for visualizing the patches feeding into the lst acordingly'''

    count = patch_count  # integer

    # creates the hsv_scale for changing the colors
    hsv_scale = np.linspace(0.1, 0.9, count + 2)
    cordinates = cordinates  # sould be a list of tuples of integers(x and y axis values)
    thickness = 5
    for i in range(0, len(cordinates)):
        color = hsv2rgb(hsv_scale[i], 1, 1)

        x, y = cordinates[i]
        start_point = (x - 16, y + 16)
        end_point = (x + 16, y - 16)

        image = cv.rectangle(image, start_point, end_point, color, thickness)

        plt.imshow(image[..., ::-1])  # to display the image without color distorntion
        plt.axis('off')
        plt.show()

        #returns the last image with axis
    return plt.imshow(image[..., ::-1])