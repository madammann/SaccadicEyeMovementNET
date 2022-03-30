'''Module imports'''

'''Library imports'''

import cv2 as cv
from matplotlib import pyplot as plt
import sys
import numpy as np
import colorsys

def load_performance_data():
    '''Loads in stored performance data.'''
    pass

def create_performance_data():
    '''Creates performance data using the model at timestamp x and data.'''

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
