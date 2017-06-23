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
params['slide_win_overlap']=0.5
params['slide_winsize']= [  (144, 144),(108, 108),(96, 96)]

def find_cars (image,params):

    dst = np.copy (image)
    all_valid_windows = []

    # iterate over previousely defined sliding windows
    for win_size in params['slide_winsize']:

        windows = slide_window(
            dst,
            x_bounds_stop=(700,None),
            y_start_stop=(400,650),
            xy_window=win_size,
            xy_overlap=params['slide_win_overlap']
        )

        valid_windows = search_windows(image, windows, svc, X_scaler, params)

        all_valid_windows.extend (valid_windows)

        dst = draw_boxes(dst, valid_windows, color=(0, 0, 1), thick=4)

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


class combine_windows():
    #Class for window combining

    def __init__(self, box):
        self.avg_box = [list(p) for p in box]
        self.detected_count = 1
        self.boxes = [box]
        self.overlap_thresh=0.3 #TBD: Parameterise

     def box_bounds(self):
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

    def check_overlap(self, box,overlap_thresh):
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
                        intersection >= overlap_thresh * area1 or
                        intersection >= overlap_thresh * area2
        ):
            return True
        else:
            return False


    def combine(self, cand_wins):
        joined = False

        for win in cand_wins:
            if self.check_overlap(win,self.overlap_thresh):
                cand_wins.remove(win)
                self.cand_wins.append(win)
                self.detected_count += 1

                self.avg_box[0][0] = min(self.avg_box[0][0], win[0][0])
                self.avg_box[0][1] = min(self.avg_box[0][1], win[0][1])
                self.avg_box[1][0] = max(self.avg_box[1][0], win[1][0])
                self.avg_box[1][1] = max(self.avg_box[1][1], win[1][1])

                joined = True

        return joined


def calc_average_boxes(cand_wins, detect_cnt_thrs):
    """Compute average boxes from specified hot boxes and returns
    average boxes with equals or higher strength
    """
    #avg_boxes = []
    cand_comb_win = []
    while len(cand_wins) > 0:
        win = cand_wins.pop(0)
        hb = combine_windows(win)
        while hb.combine(hot_boxes):
            pass
            cand_comb_win.append(hb) #hb contains all windows that are joined

    comb_wins = []
    for cand in cand_comb_win:
        if cand.detected_count >= detect_cnt_thrs:
            comb_wins.append(cand.box_bounds())
    return comb_wins


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
    hot_boxes, image_with_hot_boxes = find_cars(image,params)
    # heat map
    heat_map = get_heat_map(image, hot_boxes)

    # average boxes
    avg_boxes = calc_average_boxes(hot_boxes, 2)
    image_with_boxes = draw_boxes(image, avg_boxes, color=(0, 0, 1), thick=4)



class DetectionQueue(max_hist):

    def __init__(self):
        self.queue_max_len = max_hist  # number items to store
        self.last_wins = []

    def queue_win(self, boxes):

        if (len(self.last_boxes) > self.queue_max_len):
            tmp = self.last_wins.pop(0)

        self.last_wins.append(boxes)

    def get_queue(self):
        b = []
        for boxes in self.last_boxes:
            b.extend(boxes)
        return b


detQueue = DetectionQueue(10)


def process_image(image_orig):
    image_orig = np.copy(image_orig)
    image = image_orig.astype(np.float32) / 255

    # accumulating hot boxes over 10 last frames
    hot_boxes, image_with_hot_boxes = find_cars(image,params)
    detQueue.queue_win(hot_boxes)
    hot_boxes = detQueue.get_queue()

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
