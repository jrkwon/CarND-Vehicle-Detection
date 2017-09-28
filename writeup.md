## Writeup Template

---

**Vehicle Detection Project**

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector. 
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

**Files**

* ``project4ipython.py``: a sequential implementation to generate a Jupyter notebook later
* ``project.ipynb``: a Jupyter file for the project
* ``project.py``: a class implementation of ``project4ipython.py``

[//]: # (Image References)
[image1]: ./output_images/vehicel-non-vehicle.png
[image2]: ./output_images/vehicle-color-YCrCb-hog-0.png
[image3]: ./output_images/non-vehicle-color-YCrCb-hog-0.png
[image4]: ./output_images/vehicle-color-RGB-hog-all.png
[image5]: ./output_images/non-vehicle-color-RGB-hog-all.png
[image6]: ./output_images/vehicle-color-HLS-hog-all.png
[image7]: ./output_images/non-vehicle-color-HLS-hog-all.png
[image8]: ./output_images/vehicle-color-YCrCb-hog-all.png
[image9]: ./output_images/non-vehicle-color-YCrCb-hog-all.png
[image10]: ./output_images/find-cars-test1.png
[image11]: ./output_images/find-cars-test2.png
[image12]: ./output_images/find-cars-test3.png
[image13]: ./output_images/find-cars-test4.png
[image14]: ./output_images/find-cars-test5.png
[image15]: ./output_images/find-cars-test6.png
[image16]: ./output_images/find-car-windows-test1.png
[image17]: ./output_images/find-car-windows-test2.png
[image18]: ./output_images/find-car-windows-test3.png
[image19]: ./output_images/find-car-windows-test4.png
[image20]: ./output_images/find-car-windows-test5.png
[image21]: ./output_images/find-car-windows-test6.png
[image22]: ./output_images/heat-map-test1.png
[image23]: ./output_images/heat-map-test2.png
[image24]: ./output_images/heat-map-test3.png
[image25]: ./output_images/heat-map-test4.png
[image26]: ./output_images/heat-map-test5.png
[image27]: ./output_images/heat-map-test6.png

## [Rubric](https://review.udacity.com/#!/rubrics/513/view) Points
### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how I addressed each one.  

You're reading it!

### Histogram of Oriented Gradients (HOG)

#### 1. Explain how (and identify where in your code) you extracted HOG features from the training images.

The code for this step is contained in lines 26 through 190 of the file called `project4ipython.py`).  

I started by reading in all the `vehicle` and `non-vehicle` images.  Here is an example of one of each of the `vehicle` and `non-vehicle` classes:

![alt text][image1]

I then explored different color spaces and different `skimage.hog()` parameters (`orientations`, `pixels_per_cell`, and `cells_per_block`).  I grabbed random images from each of the two classes and displayed them to get a feel for what the `skimage.hog()` output looks like.

Here is an example using the `YCrCb` color space and HOG parameters of `orientations=8`, `pixels_per_cell=(8, 8)` and `cells_per_block=(2, 2)`:

![alt text][image2]
![alt text][image3]

#### 2. Explain how you settled on your final choice of HOG parameters.

I tried various combinations of parameters. Here are some examples of the trials.

I tested the RGB color space with a vehicle and non-vehicle image.  
![alt text][image4]
![alt text][image5]

Then the HLS color space with a vehicle and non-vehicle image was done.  
![alt text][image6]
![alt text][image7]

Finally I tested YCrCb with a vehicle and non-vehicle image.
![alt text][image8]
![alt text][image9]

After testing out with various combination of parameters, I settled on my final choice of HOG.

| HOG parameters  | Value      |
|-----------------|-------|
| color_space     | YCrCb |
| orientations    | 9     |
| pixels per cell | 16    |
| cells per block | 2     |
| hog channel     | ALL   |

#### 3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

I trained a linear SVM using the HOG parameters mentioned above with spatial and histogram features. The code for this step is contained in lines 191 through 294 of the file called `project4ipython.py`).  


### Sliding Window Search

#### 1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

I decided to search cars using the following parameters for sliding windows.

| Sliding windows parameters | Value                                   |
|----------------------------|-----------------------------------------|
| x start, x stop            | 0, image_width                          |
| y start, y stop            | image_height/2 + 40, image_height - 100 |
| window size                | 96, 96                                  |
| hog channel                | ALL                                     |

The code for this step is contained in lines 295 through 486 of the file called `project4ipython.py`).  

#### 2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

Ultimately I searched cars using YCrCb 3-channel HOG features plus spatially binned color and histograms of color in the feature vector, which provided a nice result.  Here are some example images:

![alt text][image16]
![alt text][image17]
![alt text][image18]
![alt text][image19]
![alt text][image20]
![alt text][image21]

---

### Video Implementation

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)
Here's a [link to my video result](./project_video_processed.mp4)


#### 2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

I used the test images to filter out false positives.  From the positive detections I created a heatmap and then thresholded that map to identify vehicle positions.  I then used `scipy.ndimage.measurements.label()` to identify individual blobs in the heatmap.  I then assumed each blob corresponded to a vehicle.  I constructed bounding boxes to cover the area of each blob detected.  

Here's an example result showing the bounding boxes and the heatmap from the test images given and the result of `scipy.ndimage.measurements.label()`:

![alt text][image22]
![alt text][image23]
![alt text][image24]
![alt text][image25]
![alt text][image26]
![alt text][image27]

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

In this project, I did not use the ``find_cars`` function that uses the HOG sub-sampling window search since the performance is little inferior to my ``find_cars_windows`` function that uses ``search_windows``. 

One potential problem in my pipeline is that there might be a chance to detect a car as two cars from a large car image.
