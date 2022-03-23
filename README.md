# SaccadicEyeMovementNET

## About
In this project, intended as final project of the course “Implementing Artificial Neural Networks with TensorFlow”, we implement an ANN for image classification with a different approach than the typical CNN architecture.  
Supposedly more like human vision our “SaccadicEyeMovementNet” uses small image patches from different locations to classify an image.  

## Setup
There are three prerequisites for running this code:

### Setting up the environment
The environment is specified in the environment file and can either be created using this file or manually with all the necessary pip installs.  
In the Anaconda prompt the command ``conda env create -f environment.yml`` can be used to create the environment easily.  
The environment can then be activated using ``activate SaccadicEyeMovementNet``.

### Manually downloading  annotations.json files
Furthermore, it is required to manually download annotation files for the coco dataset.  
The annotation files can be downloaded from [here](http://images.cocodataset.org/annotations/annotations_trainval2014.zip) and need to be extracted inside the root folder (the same on this README is in).  
This should result in one folder named ``annotations`` with two files called ``instances_train2014.json`` and ``instances_val2014.json``.

### Setting up the setup.txt file
In the setup.txt file we specify certain directory paths and certain parameters.
Considering the size of the dataset it may be useful to specify a different path under DataPath to another harddrive with more memory or to a specific location.
Default options are the options we used for our data and training, other options are also possible.

## Structure
TBD

## Contact
Marlon Dammann <mdammann@uni-osnabrück.de>
Nils Niehaus <nniehaus@uni-osnabrück.de>
Argha Sarker <asarker@uni-osnabrück.de>

## References
TBD
