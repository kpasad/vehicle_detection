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
params['cspace']='RGB'
params['features']=['spatial','histo','hog']
params['hog_channel']='ALL'
params['y_start_stop']=[None,None]
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

def get_hot_boxes (image,params):
    """Applies sliding windows to images
    and finds hot windows. Also returns image with all hot boxes are drawn
    Args:
        image (numpy.array): image
    Returns:
        hot_windows(list), image_with_hot_windows_drawn(numpy.array)
    """

    dst = np.copy (image)
    all_hot_windows = []

    # iterate over previousely defined sliding windows
    for x_limits, y_limits, window_size, overlap in zip (sw_x_limits, sw_y_limits, sw_window_size, sw_overlap):

        windows = slide_window(
            dst,
            x_start_stop=x_limits,
            y_start_stop=y_limits,
            xy_window=window_size,
            xy_overlap=overlap
        )

        hot_windows = search_windows(image, windows, svc, X_scaler, params)

        all_hot_windows.extend (hot_windows)

        dst = draw_boxes(dst, hot_windows, color=(0, 0, 1), thick=4)

    return all_hot_windows, dst

def get_heat_map(image, bbox_list):
    """Computes heat map of hot windows. Puts all specified
    hot windows on top of each other, so every pixel of returned image will
    contain how many hot windows covers this pixel
    Args:
        image (numpy.array): image
    Returns:
        heatmap (numpy.array) grayscale image of the same size as input image
    """

    heatmap = np.zeros_like(image[:,:,0]).astype(np.float)

    # Iterate through list of bboxes
    for box in bbox_list:
        # Add += 1 for all pixels inside each bbox
        heatmap[box[0][1]:box[1][1], box[0][0]:box[1][0]] += 1

    # Return updated heatmap
    return heatmap


class AverageHotBox():
    #Class for window combining

    def __init__(self, box):
        self.avg_box = [list(p) for p in box]
        self.detected_count = 1
        self.boxes = [box]

    def get_strength(self):
        """Returns number of joined boxes"""
        return self.detected_count

    def get_box(self):
        """Uses joined boxes information to compute
        this average box representation as hot box.
        This box has average center of all boxes and have
        size of 2 standard deviation by x and y coordinates of its points
        """
        if len(self.boxes) > 1:
            center = np.average(np.average(self.boxes, axis=1), axis=0).astype(np.int32).tolist()

            # getting all x and y coordinates of
            # all corners of joined boxes separately
            xs = np.array(self.boxes)[:, :, 0]
            ys = np.array(self.boxes)[:, :, 1]

            half_width = int(np.std(xs))
            half_height = int(np.std(ys))
            return (
                (
                    center[0] - half_width,
                    center[1] - half_height
                ), (
                    center[0] + half_width,
                    center[1] + half_height
                ))
        else:
            return self.boxes[0]

    def is_close(self, box):
        """Check wether specified box is close enough for joining
        to be close need to overlap by 30% of area of this box or the average box
        """

        # Thank you, davin
        # http://math.stackexchange.com/questions/99565/simplest-way-to-calculate-the-intersect-area-of-two-rectangles
        x11 = self.avg_box[0][0]
        y11 = self.avg_box[0][1]
        x12 = self.avg_box[1][0]
        y12 = self.avg_box[1][1]
        x21 = box[0][0]
        y21 = box[0][1]
        x22 = box[1][0]
        y22 = box[1][1]

        x_overlap = max(0, min(x12, x22) - max(x11, x21))
        y_overlap = max(0, min(y12, y22) - max(y11, y21))

        area1 = (x12 - x11) * (y12 - y11)
        area2 = (x22 - x21) * (y22 - y21)
        intersection = x_overlap * y_overlap;

        if (
                        intersection >= 0.3 * area1 or
                        intersection >= 0.3 * area2
        ):
            return True
        else:
            return False

    def join(self, boxes):
        """Join in all boxes from list of given boxes,
        removes joined boxes from input list of boxes
        """

        joined = False

        for b in boxes:
            if self.is_close(b):
                boxes.remove(b)
                self.boxes.append(b)
                self.detected_count += 1

                self.avg_box[0][0] = min(self.avg_box[0][0], b[0][0])
                self.avg_box[0][1] = min(self.avg_box[0][1], b[0][1])
                self.avg_box[1][0] = max(self.avg_box[1][0], b[1][0])
                self.avg_box[1][1] = max(self.avg_box[1][1], b[1][1])

                joined = True

        return joined


def calc_average_boxes(hot_boxes, strength):
    """Compute average boxes from specified hot boxes and returns
    average boxes with equals or higher strength
    """
    avg_boxes = []
    while len(hot_boxes) > 0:
        b = hot_boxes.pop(0)
        hb = AverageHotBox(b)
        while hb.join(hot_boxes):
            pass
        avg_boxes.append(hb) #hb contains all windows that are joined

    boxes = []
    for ab in avg_boxes:
        if ab.get_strength() >= strength:
            boxes.append(ab.get_box())
    return boxes


X_scaler,svc =pk.load(open(basepath+'/classifier.pk','rb'))

test_images = []
test_images_titles = []

for impath in glob.glob(basepath+'/test_images/test*.jpg'):
    print('image {}'.format(impath))
    image_orig = mpimg.imread(impath)

    # Uncomment the following line if you extracted training
    # data from .png images (scaled 0 to 1 by mpimg) and the
    # image you are searching is a .jpg (scaled 0 to 255)
    image = image_orig.astype(np.float32) / 255

    # hot boxes
    hot_boxes, image_with_hot_boxes = get_hot_boxes(image,params)
    # heat map
    heat_map = get_heat_map(image, hot_boxes)

    # average boxes
    avg_boxes = calc_average_boxes(hot_boxes, 2)
    image_with_boxes = draw_boxes(image, avg_boxes, color=(0, 0, 1), thick=4)

    test_images.append(image_with_hot_boxes)
    test_images.append(heat_map)
    test_images.append(image_with_boxes)

    test_images_titles.extend(['', '', ''])

test_images_titles[0] = 'hot boxes'
test_images_titles[1] = 'heat map'
test_images_titles[2] = 'average boxes'

#img_grid(test_images,(6,3))


# in video I use information from multiple frames to
# make average boxes more robust and filter false positives
# I accumulate all hot boxes from last several frames and use them
# for calculating average boxes

class LastHotBoxesQueue():
    """Class for accumulation of hot boxes from last 10 frames
    """

    def __init__(self):
        self.queue_max_len = 10  # number items to store
        self.last_boxes = []

    def put_hot_boxes(self, boxes):
        """Put frame hot boxes
        """
        if (len(self.last_boxes) > self.queue_max_len):
            tmp = self.last_boxes.pop(0)

        self.last_boxes.append(boxes)

    def get_hot_boxes(self):
        """Get last 10 frames hot boxes
        """
        b = []
        for boxes in self.last_boxes:
            b.extend(boxes)
        return b


last_hot_boxes = LastHotBoxesQueue()


def process_image(image_orig):
    image_orig = np.copy(image_orig)
    image = image_orig.astype(np.float32) / 255

    # accumulating hot boxes over 10 last frames
    hot_boxes, image_with_hot_boxes = get_hot_boxes(image,params)
    last_hot_boxes.put_hot_boxes(hot_boxes)
    hot_boxes = last_hot_boxes.get_hot_boxes()

    # calculating average boxes and use strong ones
    # need to tune strength on particular classifer
    avg_boxes = calc_average_boxes(hot_boxes, 20)
    image_with_boxes = draw_boxes(image, avg_boxes, color=(0, 0, 1), thick=4)

    return image_with_boxes * 255


from moviepy.editor import VideoFileClip


def process_video(input_path, output_path):
    clip = VideoFileClip(input_path)

    result = clip.fl_image(process_image)
    result.write_videofile(output_path)


# select video to operate on
# process_video ('test_video.mp4', 'test_video_result.mp4')
process_video('project_video.mp4', 'project_video_result.mp4')
