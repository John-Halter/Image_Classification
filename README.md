# Image Classification
#### A butterfly image classifier using Tensorflow, Keras and Pycharm
#### Created by John Halter

## Description of Project
___
The following project uses a convolutional neural network to classify some of the most common butterflies in Colorado.
The reasons I wanted to make this type of project was to gain experience working with supervised learning models.
Another goal of mine was to create my own dataset from scratch to really comprehend the models output.
Using this personally created dataset, I wanted  to use Keras and Tensorflow to create a
convolutional neural network that could classify the following butterflies native to Colorado:
- Colorado Hairstreak
- Common Buckeye
- Monarch
- Red Admiral
- Rocky Mountain Parnassian
- Two-Tailed Swallowtail

## Software Used
___
- Python 3.10
- Pycharm 2021.3

## Project Structure:

### 1. Dataset Creation
___
The dataset was created using a module that scrapped bing images named 
[icrawler](https://icrawler.readthedocs.io/en/latest/builtin.html). This module has several parameters and with these
the different butterfly names and number of photos to extract would be used as input. 
The package then downloads all available pictures from Bing images. The function where this occurs is called 
*download_pictures* in the *dataset_gather* located in the *dataset_creation* directory.
The problem with this is it also downloads unwanted photos. Another function was created to address this issue.
This function called *verify_pictures* in the code is to go through each individual image and keep or delete the 
current displayed photo. After verifying all the images The next step is to create the model and train it.
### 2. Model Creation
___
For the CNN model to work it needs a training and testing dataset. After initial testing the dataset that was being used 
was overfitting. To address this, data augmentation would be used to create a better dataset to work with. 
Using keras and its *Image_Generator* functionality, the datasets were both augmented and the total number of images
was increased. With a better dataset the model would improve and be adjusted when necessary. 
The model itself was created also using keras Sequential model. The model is outlined below:

<p align="center">
  <img src="C:\Users\Johnny\PycharmProjects\Image_Classification\images\cnn_outline_model.png" width="375" />
</p>

The model rescales the images down to all be equal. Then it uses multiple convolutional 2D layers and pooling
them as well. After testing multiple models this one was the final result.
As this model uses a lot of trainable parameters because the image's perspective has more variety.
A dropout layer is used as one of the final layers to prevent even more overfitting. As for the last couple of layers,
the flatten and dense layers are to flatten the input so the dense layer can make the final classification of the images.
### 3. Results
___
After tweaking and making several iterations of the model above the final results are shown below:
<p align="center">
  <img src="C:\Users\Johnny\PycharmProjects\Image_Classification\images\model_accuracy_loss_plot.png" width="550" />
</p>
From the image shown the model reached an average accuracy of ~85% over 40 iterations. The reason for so many iterations is to 
compensate for the lack of data. The regression lines help clear some of the variability of the inital plots.
From these plots it still looks to seem that they are trending upward. Increasing the number of iterations can show that

<p align="center">
  <img src="C:\Users\Johnny\PycharmProjects\Image_Classification\images\model_accuracy_loss_plot50.png" width="550" />
</p>

With the number of epochs increased to 50 we see more of a plateau in the average accuracy of ~90%. 
Meaning that 40-50 epochs is probably the most effective number for accuracy with this dataset. 


### How To Use Project
___
To use this project all functionality is stored in main. Beginning with the first condition *download_images*.
If you want to run it properly start with this condition by itself then do the same for the other conditions.
If done right the pictures will be downloaded, then the user will have to verify the images, the dataset will be split,
Then the model will be run. The file *CONSTANTS* is used to adjust several parameters used throughout the code. 

### Improvements To Be Made
___

- [ ] Increase number of images for the dataset
- [ ] Add more visualizations for multiple iterations of running model
- [ ] Change split_train_test to keep images in initial directories too