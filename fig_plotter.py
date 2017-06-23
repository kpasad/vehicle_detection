import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import cv2
import glob
import time
from sklearn.svm import LinearSVC,SVC
from sklearn.preprocessing import StandardScaler
from skimage.feature import hog
from veh_det_lib import *
# NOTE: the next import is only valid for scikit-learn version <= 0.17
# for scikit-learn >= 0.18 use:
# from sklearn.model_selection import train_test_split
from sklearn.cross_validation import train_test_split
import pickle as pk
import random
from mpl_toolkits.axes_grid1 import ImageGrid


def img_grid(image, nrows_ncols,title=[]):
    #n = int(np.ceil(np.sqrt(len(image))))
    nrows = nrows_ncols[0]
    ncols = nrows_ncols[1]
    if len(image)> nrows*ncols:
        nrows=int(np.ceil(len(image)/ncols))
    fig = plt.figure(1, (40., 40.))
    grid = ImageGrid(fig, 111,  # similar to subplot(111)
                     (nrows,ncols),  # creates 2x2 grid of axes
                     axes_pad=0.1,  # pad between axes in inch.
                     )

    for i in range(len(image)):
        # grid[i].imshow(image[i],cmap='gray',animated=True)  # The AxesGrid object work as a list of axes.
        grid[i].imshow(image[i], cmap='gray')  # The AxesGrid object work as a list of axes.
        # grid[i].set_title(title[i])
        grid[i].axis('off')
    plt.show()


# Define a function to extract features from a single image window
# This function is very similar to extract_features()
# just for a single image rather than list of images
def single_img_features(img,params):
    cspace=params['cspace']
    
    #1) Define an empty list to receive features
    img_features = []
    #2) Apply color conversion if other than 'RGB'
    if color_space != 'RGB':
        if color_space == 'HSV':
            feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
        elif color_space == 'LUV':
            feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2LUV)
        elif color_space == 'HLS':
            feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
        elif color_space == 'YUV':
            feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)
        elif color_space == 'YCrCb':
            feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb)
    else: feature_image = np.copy(img)      
    #3) Compute spatial features if flag is set
    if 'spatial' in params['features'] :
        spatial_features = bin_spatial(feature_image, params)
        #4) Append features to list
        img_features.append(spatial_features)
    #5) Compute histogram features if flag is set
    if 'histo' in params['features']:
        hist_features = color_hist(feature_image, params)
        #6) Append features to list
        img_features.append(hist_features)
    #7) Compute HOG features if flag is set
    if 'hog' in params['features']:
        if params['hog_channel'] == 'ALL':
            hog_features = []
            for channel in range(feature_image.shape[2]):
                hog_features.extend(get_hog_features(feature_image[:,:,channel], params))      
        else:
            hog_features = get_hog_features(feature_image[:,:,hog_channel], params)
        #8) Append features to list
        img_features.append(hog_features)

    #9) Return concatenated array of features
    return np.concatenate(img_features)

# Define a function you will pass an image 
# and the list of windows to be searched (output of slide_windows())
def search_windows(img, windows, clf, scaler, color_space='RGB', 
                    spatial_size=(32, 32), hist_bins=32, 
                    hist_range=(0, 256), orient=9, 
                    pix_per_cell=8, cell_per_block=2, 
                    hog_channel=0, spatial_feat=True, 
                    hist_feat=True, hog_feat=True):

    #1) Create an empty list to receive positive detection windows
    on_windows = []
    #2) Iterate over all windows in the list
    for window in windows:
        #3) Extract the test window from original image
        test_img = cv2.resize(img[window[0][1]:window[1][1], window[0][0]:window[1][0]], (64, 64))      
        #4) Extract features for that window using single_img_features()
        features = single_img_features(test_img, params)
        #5) Scale extracted features to be fed to classifier
        test_features = scaler.transform(np.array(features).reshape(1, -1))
        #6) Predict using your classifier
        prediction = clf.predict(test_features)
        #7) If positive (prediction == 1) then save the window
        if prediction == 1:
            on_windows.append(window)
    #8) Return windows for positive detections
    return on_windows
    
    
    
    

basepath = r'C:\Users\kpasad\Dropbox\ML\project\sdc\CarND-vehicle-detection'
#basepath = r'C:\Users\kpasad\mydata\sdc\vehicle_det_data'
basepath = basepath.replace('\\','/')
cars = glob.glob(basepath+'/vehicles/*/*.png')
notcars = glob.glob(basepath+'/non-vehicles/*/*.png')

print('number of cars is {}'.format(len(cars)))
print('number of non cars is {}'.format(len(notcars)))
'''
cars_nocars=[]
for i in range(0,4):
    cars_nocars.append(cv2.imread(cars[random.randint(0,100)]))
for i in range(0,4):
    cars_nocars.append(cv2.imread(notcars[random.randint(0,100)]))

img_grid(cars_nocars,(2,4))
'''

params={}
params['n_orient_bins']=8
params['pixels_per_cell']=(8,8)
params['cell_per_block']=(2,2)
params['vis'] =True
params['collapse_feat']=True
params['spatial_bin_size']=(16,16)
params['color_hist_nbins']=16
params['color_hist_bins_range']=(0,256)
params['cspace']='RGB'
params['features']=['spatial','histo','hog']
params['hog_channel']='ALL'
'''
fix,ax = plt.subplots(3,2)
for i in range(0,3):
    img = cv2.imread(cars[random.randint(0,100)])
    ax[i][0].imshow(img, cmap='gray')  # The AxesGrid object work as a list of axes.
    img_hsv=cv2.cvtColor(img,cv2.COLOR_RGB2HLS)
    hog_feats,hog_img=get_hog_features(img_hsv[:,:,2],params)
    ax[i][1].imshow(hog_img, cmap='gray')  # The AxesGrid object work as a list of axes.
    ax[i][0].axis('off')
    ax[i][1].axis('off')
    ax[i][0].set_adjustable('box-forced')
    ax[i][1].set_adjustable('box-forced')
#plt.tight_layout()
#plt.show()
'''
from mpl_toolkits.axes_grid1 import ImageGrid
'''
fig = plt.figure(1, (40., 40.))
grid = ImageGrid(fig, 111,  # similar to subplot(111)
                 nrows_ncols=(3, 2),  # creates 2x2 grid of axes
                 axes_pad=0.1,  # pad between axes in inch.
                 )

for i in range(0,3):
    print(i)
    img = cv2.imread(notcars[random.randint(0,100)])
    grid[2*i].imshow(img, cmap='gray')  # The AxesGrid object work as a list of axes.
    img_hsv=cv2.cvtColor(img,cv2.COLOR_RGB2HLS)
    hog_feats,hog_img=get_hog_features(img_hsv[:,:,2],params)
    grid[2 * i+1].imshow(hog_img, cmap='gray')  # The AxesGrid object work as a list of axes.
'''

image = mpimg.imread('test_images/test1.jpg')
window_img = np.copy(image)

sw_x_limits = [
    [None, None],
    [32, None],
    [412, 1280]
]

sw_y_limits = [
    [400, 640],
    [400, 600],
    [390, 540]
]

sw_window_size = [
    (128, 128),
    (96, 96),
    (80, 80)
]

sw_overlap = [
    (0.5, 0.5),
    (0.5, 0.5),
    (0.5, 0.5)
]

# create sliding windows
windows = slide_window(image, x_start_stop=sw_x_limits[0], y_start_stop=sw_y_limits[0],
                    xy_window=sw_window_size[0], xy_overlap=sw_overlap[0])

windows2 = slide_window(image, x_start_stop=sw_x_limits[1], y_start_stop=sw_y_limits[1],
                    xy_window=sw_window_size[1], xy_overlap=sw_overlap[1])

windows3 = slide_window(image, x_start_stop=sw_x_limits[2], y_start_stop=sw_y_limits[2],
                    xy_window=sw_window_size[2], xy_overlap=sw_overlap[2])

# show sliding windows
sliding_windows = []
sliding_windows.append (draw_boxes(np.copy(image), windows, color=(0, 0, 0), thick=4))
sliding_windows.append (draw_boxes(np.copy(image), windows2, color=(0, 0, 0), thick=4))
sliding_windows.append (draw_boxes(np.copy(image), windows3, color=(0, 0, 0), thick=4))

# drawing one of sliding windows in blue
sliding_windows [0] = draw_boxes (sliding_windows [0], [windows[9]], color=(0, 0, 255), thick=8)
sliding_windows [1] = draw_boxes (sliding_windows [1], [windows2[12]], color=(0, 0, 255), thick=8)
sliding_windows [2] = draw_boxes (sliding_windows [2], [windows3[5]], color=(0, 0, 255), thick=8)

sw_titles = [
    '128 x 128 windows',
    '96 x 96',
    '80 x 80'
]
fig = plt.figure(1, (1., 3.))
grid = ImageGrid(fig, 111,  # similar to subplot(111)
                 nrows_ncols=(3, 1),  # creates 2x2 grid of axes
                 axes_pad=0.1,  # pad between axes in inch.
                 )
grid[0].imshow(draw_boxes(np.copy(image), windows, color=(0, 0, 0), thick=4))
grid[1].imshow(draw_boxes(np.copy(image), windows2, color=(0, 0, 0), thick=4))
grid[2].imshow(draw_boxes(np.copy(image), windows3, color=(0, 0, 0), thick=4))