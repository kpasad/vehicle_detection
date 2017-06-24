import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import cv2
import glob
import time
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
from skimage.feature import hog
from veh_det_lib import *
import pickle as pk
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
        grid[i].imshow(image[i], cmap='gray')  # The AxesGrid object work as a list of axes.
        #grid[i].set_title(title[i])

    plt.show()

# Define a function to extract features from a single image window
# This function is very similar to extract_features()
# just for a single image rather than list of images
def single_img_features(img,params):
    color_space=params['cspace']

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

def search_windows(img, windows, clf, scaler, params):
    #1) Create an empty list to receive positive detection windows

    color_space = params['color_space']
    spatial_size = params['spatial_bin_size']
    hist_bins = params['color_hist_nbins'],
    hist_range = params['color_hist_bins_range']
    orient = params['n_orient_bins']
    pix_per_cell = params['pixels_per_cell']
    cell_per_block = params['cell_per_block']
    hog_channel = params['hog_channel']
    if 'spatial' in params['features']:
        spatial_feat = True
    if 'histo' in params['features']:
        hist_feat = True
    if 'hog' in params['features']:
        hog_feat = True




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
basepath = basepath.replace('\\','/')

params={}
params['color_space']='RGB'
params['n_orient_bins']=8
params['pixels_per_cell']=(16,16)
params['cell_per_block']=(1,1)
params['vis'] =False
params['collapse_feat']=True
params['spatial_bin_size']=(16,16)
params['color_hist_nbins']=16
params['color_hist_bins_range']=(0,256)
params['cspace']='HLS'
params['features']=['spatial','histo','hog']
params['hog_channel']='ALL'
params['y_start_stop']=[None,None]
params['slide_win_overlap']=(0.5,0.5)
params['slide_winsize']= [  (128, 128),(96, 96),(80, 80)]

def find_cars (image,params):

    dst = np.copy (image)
    all_valid_windows = []

    # iterate over previousely defined sliding windows
    for win_size in params['slide_winsize']:

        windows = slide_window(
            dst,
            x_start_stop=[700,None],
            y_start_stop=[400,650],
            xy_window=win_size,
            xy_overlap=params['slide_win_overlap']
        )

        valid_windows = search_windows(image, windows, svc, X_scaler, params)

        all_valid_windows.extend (valid_windows)

        dst = draw_boxes(dst, valid_windows, color=(0, 0, 1), thick=4)

    return all_valid_windows, dst

def add_heat(heatmap, bbox_list):
    # Iterate through list of bboxes
    for box in bbox_list:
        # Add += 1 for all pixels inside each bbox
        # Assuming each "box" takes the form ((x1, y1), (x2, y2))
        heatmap[box[0][1]:box[1][1], box[0][0]:box[1][0]] += 1

    # Return updated heatmap
    return heatmap# Iterate through list of bboxes

def apply_threshold(heatmap, threshold):
    # Zero out pixels below the threshold
    heatmap[heatmap <= threshold] = 0
    # Return thresholded map
    return heatmap

def draw_labeled_bboxes(img, labels):
    # Iterate through all detected cars
    for car_number in range(1, labels[1]+1):
        # Find pixels with each car_number label value
        nonzero = (labels[0] == car_number).nonzero()
        # Identify x and y values of those pixels
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        # Define a bounding box based on min/max x and y
        bbox = ((np.min(nonzerox), np.min(nonzeroy)), (np.max(nonzerox), np.max(nonzeroy)))
        # Draw the box on the image
        cv2.rectangle(img, bbox[0], bbox[1], (0,0,255), 6)
    # Return the image
    return img

X_scaler,svc =pk.load(open(basepath+'/classifier.pk','rb'))

test_images = []
test_images_titles = []

class DetectionQueue():
    def __init__(self,max_hist):
        self.queue_max_len = max_hist  # number items to store
        self.last_wins = []

    def queue_win(self, boxes):

        if (len(self.last_wins) > self.queue_max_len):
            tmp = self.last_wins.pop(0)

        self.last_wins.append(boxes)

    def get_queue(self):
        b = []
        for boxes in self.last_wins:
            b.extend(boxes)
        return b


detQueue = DetectionQueue(20)



from scipy.ndimage.measurements import label

def process_image(image_orig):
    image_orig = np.copy(image_orig)
    image = image_orig.astype(np.float32) / 255
    # find cars in this frame.
    hot_boxes, image_with_hot_boxes = find_cars(image, params)
    #Add the detection to the Queue
    detQueue.queue_win(hot_boxes)
    q_hot_boxes = detQueue.get_queue()

    heat = np.zeros_like(image[:, :, 0]).astype(np.float)
    heatmap = add_heat(heat, q_hot_boxes)
    thrs_heatmap = apply_threshold(heatmap,18)

    # Find final boxes from heatmap using label function
    labels = label(thrs_heatmap)
    draw_img = draw_labeled_bboxes(np.copy(image_orig), labels)
    #plt.imshow(draw_img)
    return draw_img


from moviepy.editor import VideoFileClip


def process_video(input_path, output_path):
    #clip = VideoFileClip(input_path).subclip(23,28)
    clip = VideoFileClip(input_path)
    result = clip.fl_image(process_image)
    result.write_videofile(output_path)


# select video to operate on
process_video('project_video.mp4', 'project_video_result_v2.mp4')
