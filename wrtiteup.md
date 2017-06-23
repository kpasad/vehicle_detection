
# Vehicle Detection Project

[//]: # (Image References)
[image1]: ./images/cars_nocars.png
[image2a]: ./images/cars_hog.png
[image2b]: ./images/cars_hog_8x8.png
[image2c]: ./images/nocars_hog.png
[image3]: ./images/windows.png
[image4]: ./images/pipeline.png
[image5]: ./examples/bboxes_and_heat.png
[image6]: ./examples/labels_map.png
[image7]: ./examples/output_bboxes.png
[video1]: ./project_video.mp4


## Introduction

The goal of Vehicle detection project is to create a processing pipeline that can detect  presense of vehicle in the adjacent lanes of an autonomous vehicle. In general the pipeline is composed of:
1. A classifier trained to claasify images as belonging to cars or no- cars in a tightly cropped image
2. A windowing scheme that crops a large image into smaller windows and uses the classifier to detect the presense or absense of cars in the window
3. A tracking and combining scheme, that serves to minimise false alarms and combine multiple windows into a single window.  

### Classifier and features
The base classifier uses the combination of following features:

*  Histogram of Colors
*  Spatial binning 
*  Histogram of Gradients

All three feature extraction schemes are a part of the library file, veh_lib_det.py and are contained in their name-sake functions.

### Training data set:
The training data set are made available by Udacity [here ](https://github.com/udacity/CarND-Vehicle-Detection). The dataset consists of a series of tightly cropped images, 64 by 64 pixels each. The data set is balanced and contains about 8700 images for each category. Below are several example images
 ![cars vs not cars][image1]  


### Histogram of Oriented Gradients (HOG)
HOG is computed using skimage library function called `skimage.hog()`. This function are called from get_hog_features() in the library file veh_det_lib.py.  
Following are example HOG features for cars and non cars for various color spaces. HOG features are dervied from the Saturation dimension of a HLS color space. Saturation is chosen because vehicles are markedly saturated in their colors as compared to the background (road/sky). Of all the HOG parameters the number of pixels per cell, unsurprisingly, has the  biggest visual impact.
Below is the HOG feature for cars, with 16 pixels per cell.
![alt text][image2a]
When the number of pixels is reduced to 8 per cell, we have 2x more cells and the hog feature 'resolution' improves. An outline representing the general shape of the vehicle is visible.
![alt text][image2b]
In case of no cars, the HOG is not very discriminative. See below
![alt text][image2c]

### Training the classifier
The first classifier was a linear SVM. The cross validation error with linear SVM was ~97%. Despite the low classification error rate, a false alarm was visible in the test images. When tried on the video sequence, the false alarms and missed detections were readily apparent. There were two options on reducing the false alarm rate. Improving the sliding window processing pipeline or improving the classification itself. I chose the later option as it was a quick test. The SVM kernel was switched to a non-linear RBF kernel. The classification error improved to ~99%. The seemingly marginal difference had a significant influence on the false alarm and quality of video.  

### Sliding Window Search
Three window sizes were used corresponding to near field, mid field and far field with respect to the camera. The window search was limited to a trapezoid in front of the camera. The window span for each of the three sizes are 128,96, and 80. They were chosen over several iterations and visual inspection. The overlap between the windows was 50%. The function slide_window() in veh_det_lib.py generates the coordinates of all windows of a given size and given overlap. It is called thrice for three window sizes
See the images

![alt text][image3]

### Pipeline
Figure below shows the processing pipeline for the test images. The first row show overlap of multiple windows. The middle row is the heat map. The final row shows the result of window combining. Image #2 has a false alarm that is removed eventually. The third image has a missed detection. 
![alt text][image4]


In any given frame, multiple windows will generate a positive  classification for the same vehicle. The window joining algorithm looks for overlapping windows, where the overlap is defined as the union of two windows. If the overlap greater than a threshold, the windows are joined. Joining entails using the maximum coordinates along each axis in both direction, where maximum is from the center of image.
The number of windows joined is greater than a programmable threshold, the windows in the set constitute a car. This mechanism also provides guard against false alarms. The windows are joined over multiple frames to create a more robust solution.

### Video Implementation

Here's a [link to my video result](https://www.dropbox.com/s/sr8xb2q5gdrwlqr/project_video_result.mp4?dl=0)

### Discussion

The SVM is trained on images of cars or no cars. During test, however, the data consist of partial cars. So the test data is different. The HOG features makes it somewhat invariant to the location of the car in the image. A much robust classifier can be created if images of partial cars, akin to  a sliding window, are used for training. Such a data set need not be hand annotated, but could be automated. 

