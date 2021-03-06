import argparse
import base64
from datetime import datetime
import os
import shutil
import copy

import numpy as np
import matplotlib.pyplot as plt
import cv2
import glob
import math
from moviepy.editor import VideoFileClip

import logging
logging.basicConfig(level=logging.INFO)


parser = argparse.ArgumentParser(description='Advanced lane finding project.')
parser.add_argument('-c', '--cb_img_dir', default="./camera_cal", help='Chess-board images dir.')
parser.add_argument('-nx', '--nx', default=9, help='Number of chessboard corners in the x-dimension')
parser.add_argument('-ny', '--ny', default=6, help='Number of chessboard corners in the y-dimension')
parser.add_argument('-f', '--img_format', default="jpg", help='Format of camera calibration images')
parser.add_argument('-p', '--img_prefix', default="calibration", help='Prefix for calibration images (used in glob)')
parser.add_argument('-t', '--test_img_dir', default="test_images", help='Directory for test images')

LEFT = 0
RIGHT = 1
DEBUG = 1
VERBOSE = 0
mtx = []
dist = []
image_prefix = ""
image_format = "jpg"
img_dir = "./test_images"
img_index = 0
objp = np.zeros(( 2 * 2,2), np.float32)
objpoints = []
imgpoints = []

use_history = False
hist_length = 7
cur_len = 0
weights = []
left_bottom_hist = []
left_top_hist = []
right_bottom_hist = []
right_top_hist = []

LEFT = 0
RIGHT = 1
DEBUG = 0

def reset_history():
    '''
    Reset history. Call this before processing a new video.
    '''
    global left_bottom_hist
    global left_top_hist
    global right_bottom_hist
    global right_top_hist
    cur_len = 0
    left_bottom_hist[:] = []
    left_top_hist[:] = []
    right_bottom_hist[:] = []
    right_top_hist[:] = []

def weighted_avg(values, weights):
    w_a = 0
    for i in range(len(values)):
        w_a += values[i] * weights[i]
    return int(w_a)
    
def weighted_avg_hist(cur_val, val_hist, val_wt):
    ret_val = cur_val
    global cur_len
    idx = len(val_hist) - 1
    for i in range(cur_len):
        ret_val += val_hist[idx - i] * val_wt[i]
    return ret_val


def weighted_img(img, initial_img, alpha=0.8, beta=1., ro=0.):
    """
    `img` is the output of the hough_lines(), An image with lines drawn on it.
    Should be a blank image (all black) with lines drawn on it.
    
    `initial_img` should be the image before any processing.
    
    The result image is computed as follows:
    
    initial_img * α + img * β + λ
    NOTE: initial_img and img must be the same shape!
    """
    logging.debug("Final Img shape: %s" %str(img.shape))
    logging.debug("Orig  Img shape: %s" %str(initial_img.shape))
    return cv2.addWeighted(initial_img, alpha, img, beta, ro)

def int_mid(coords):
    '''
    Get the median of  a set of points.
    '''
    med = np.median(coords)
    if not math.isnan(med):
        return int(med)
    else:
        return 0

def prune_lines(bottom, top, weights, side, width):
    global VERBOSE
    index = 0
    for (t, b) in zip(top, bottom):
        remove = False
        if (side == LEFT):
            if (t < b) or (t > width / 2) or (b > width / 2):
                remove = True
        else:
            # Side is right
            if (t > b) or (t < width / 2) or (b < width / 2):
                remove = True
        if (remove):
            if (VERBOSE):
                logging.debug ("Removing noisy line: side=%d, width=%d (%d, %d)" %(side, width, t, b))
            top.pop(index)
            bottom.pop(index)
            weights.pop(index)
        else:
            index += 1

def get_x_at(y, line):
    """
    Calclulate m and c in y = mx + c, given x1, y1 and x2, y2
    Return the x intercept given y
    """
    x1, y1, x2, y2 = [float(a) for a in line]
    
    
    # Vertical line case, return one of the x's
    if (x1 == x2):
        return int(x1)
    
    # Calculate slope
    m = (y2 - y1) / (x2 - x1)
    
    # Check if line is horizontal before the next division
    if (m == 0):
        return y
    
    # Calculate the intercept, and x value given y.
    c = y2 - m * x2
    return int((y - c) / m)

def weights(in_list):
    '''
    Get the weighted coordinate of a line. 
    NOTE: this is not used. See my notes. Used in one of the intermediate version of my algorithm.
    '''
    total_wt = sum(in_list)
    if (total_wt == 0):
        return in_list
    return [(w / total_wt) for w in in_list]

def draw_lines(img, lines, color=[0, 0, 255], thickness=2):

   
    # First sort the lines into left and right.
    img_center_x = img.shape[1] / 2
    
    # A line is (x1, y1, x2, y2). 
    # We're just checking if x1 falls to left of image center.
    left_lines = [line[0] for line in lines if (line[0][0] < img_center_x)]#get_left_lines(lines, img_center_x)
    left_line_weights = weights([len(line) for line in left_lines])
    
    
    # A line is (x1, y1, x2, y2). 
    # We're just checking if x1 falls to right of image center.
    right_lines = [line[0] for line in lines if (not line[0][0] < img_center_x)] #get_right_lines(lines, img_center_x)
    right_line_weights = weights([len(line) for line in right_lines])
    
    # Find  bottom and top intercept for left lines
    left_bottom_xs = [get_x_at(img.shape[0], line) for line in left_lines]
    left_top_xs = [get_x_at(img.shape[0] * 0.62, line) for line in left_lines]
    
    prune_lines(left_bottom_xs, left_top_xs, left_line_weights, LEFT, img.shape[1])
    
    # Find bottom and top intercept for right lines
    right_bottom_xs = [get_x_at(img.shape[0], line) for line in right_lines]
    right_top_xs = [get_x_at(img.shape[0] * 0.62, line) for line in right_lines]
    
    prune_lines(right_bottom_xs, right_top_xs, right_line_weights, RIGHT, img.shape[1])
    
    # Get the median of the intercepts
    left_bottom_x = int_mid(left_bottom_xs)
    left_top_x = int_mid(left_top_xs)
    
    right_bottom_x = int_mid(right_bottom_xs)
    right_top_x = int_mid(right_top_xs)
   
    # Weighted average scheme. Did NOT solve the bogus lines issue.
    # Median performed far better.
    
    global use_history, weights, left_bottom_hist, left_top_hist, right_bottom_hist, right_top_hist

    if (use_history):
        # Sanity check. If sanity check fails, and history is enabled, just use the previous frame's value.
        if (left_top_x > right_top_x or left_bottom_x > right_bottom_x 
            or left_bottom_x <= 0 or right_bottom_x >= img.shape[1]-1 
            or left_top_x < left_bottom_x or right_top_x > right_bottom_x):
            # Sanity check failed. Use previous frame's values  available
            #logging.warning ("Sanity check failed: (lt, lb, rt, rb) = (%d, %d, %d, %d). Using history (%d, %d, %d, %d)"
            print ("Sanity check failed: (lt, lb, rt, rb) = (%d, %d, %d, %d). Using history (%d, %d, %d, %d)"
                   %(left_top_x, left_bottom_x, right_top_x, right_bottom_x, 
                     left_top_hist[-1], left_bottom_hist[-1], right_top_hist[-1], right_bottom_hist[-1]))
            left_top_x = left_top_hist[-1]
            left_bottom_x = left_bottom_hist[-1]
            right_top_x = right_top_hist[-1]
            right_bottom_x = right_bottom_hist[-1]
        else:
            # Use a weighted average.
            left_bottom_x = weighted_avg_hist(left_bottom_x, left_bottom_hist, weights) 
            left_top_x = weighted_avg_hist(left_top_x, left_top_hist, weights) 
            right_bottom_x = weighted_avg_hist(right_bottom_x, right_bottom_hist, weights) 
            right_top_x = weighted_avg_hist(right_top_x, right_top_hist, weights) 
            
        left_bottom_hist.append(left_bottom_x)
        left_top_hist.append(left_top_x)
        right_bottom_hist.append(right_bottom_x)
        right_top_hist.append(right_top_x)
        
        
    
    
    #for line in lines:
    #    for x1,y1,x2,y2 in line:
    line_img = copy.deepcopy(img)
    cv2.line(line_img, (left_bottom_x, img.shape[0]), (left_top_x, int(img.shape[0] * 0.62)), color, 8)
    cv2.line(line_img, (right_bottom_x, img.shape[0]), (right_top_x, int(img.shape[0] * 0.62)), color, 8)
    poly_to_fill = [(left_bottom_x, img.shape[0]), # left-bottom
                    (left_top_x, int(img.shape[0] * 0.62)), # left-top
                    (right_top_x, int(img.shape[0] * 0.62)), # right-top
                    (right_bottom_x, img.shape[0])] # right-bottom
    
    cv2.fillPoly(img, np.int32( [ poly_to_fill ]), (0, 64, 0) )
    

    poly_to_ret = [ (img.shape[0], left_bottom_x), # left-bottom
                    (int(img.shape[0] * 0.62), left_top_x), # left-top
                    (int(img.shape[0] * 0.62), right_top_x), # right-top
                    (img.shape[0], right_bottom_x)] # right-bottom   

    poly_to_ret = [poly_to_fill[1], poly_to_fill[2], poly_to_fill[0], poly_to_fill[3]]

    return poly_to_ret, img

def region_of_interest(img, vertices):
    """ 
    Applies an image mask.
    
    Only keeps the region of the image defined by the polygon
    formed from `vertices`. The rest of the image is set to black.
    """
    #defining a blank mask to start with
    mask = np.zeros_like(img)   
    
    #defining a 3 channel or 1 channel color to fill the mask with depending on the input image
    if len(img.shape) > 2:
        channel_count = img.shape[2]  # i.e. 3 or 4 depending on your image
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255
        
    #filling pixels inside the polygon defined by "vertices" with the fill color    
    cv2.fillPoly(mask, np.int32([vertices]),128)
    
    #returning the image only where mask pixels are nonzero
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image

def hough_lines(img, rho, theta, threshold, min_line_len, max_line_gap):
    """
    `img` should be the output of a Canny transform.
        
    Returns an image with hough lines drawn.
    """
    lines = cv2.HoughLinesP(img, rho, theta, threshold, np.array([]), minLineLength=min_line_len, maxLineGap=max_line_gap)
    line_img = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
    poly, line_img = draw_lines(line_img, lines)
    return poly, line_img


def draw_lines_on_image(orig_image, image, img_dir, img_suffix, img_format, is_rgb_lines=False):
    ## Copy-paste from quiz.
    gray = image # cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
   

    ret, bin_thresh = cv2.threshold(gray, 160, 255, cv2.THRESH_BINARY)
    
    # Define a kernel size and apply Gaussian smoothing
    blur_gray = cv2.GaussianBlur(bin_thresh, (5, 5), 0)
    

    # Define our parameters for Canny and apply
    low_threshold = 100
    high_threshold = 150
    edges = cv2.Canny(blur_gray, low_threshold, high_threshold)

    # Next we'll create a masked edges image using cv2.fillPoly()
    mask = np.zeros_like(edges)   
    ignore_mask_color = 255   

    # This time we are defining a four sided polygon to mask
    imshape = image.shape
    vertices = np.array([[(imshape[1] *0.4, imshape[0] * 0.65),(imshape[1] * 0.6, imshape[0] * 0.65),            (imshape[1] * 1. , imshape[0] * 0.85), (imshape[1] * 0.0, imshape[0]* 0.85)]], dtype=np.int32)
    masked_edges = region_of_interest(edges, vertices)
    

    # Define the Hough transform parameters
    # Make a blank the same size as our image to draw on
    rho = 1 # distance resolution in pixels of the Hough grid
    theta = np.pi/180 # angular resolution in radians of the Hough grid
    threshold = 22     # minimum number of votes (intersections in Hough grid cell)
    min_line_length = 50 #minimum number of pixels making up a line
    max_line_gap = 350    # maximum gap in pixels between connectable line segments
    line_image = np.copy(image)*0 # creating a blank to draw lines on

    # Run Hough on edge detected image
    # Output "lines" is an array containing endpoints of detected line segments
    poly, line_image = hough_lines(masked_edges, rho, theta, threshold, min_line_length, max_line_gap)

    # cv2.imwrite(img_dir + "/__green__" + img_suffix + "." +img_format, line_image)

    
    # Iterate over the output "lines" and draw lines on a blank image

    # Create a "color" binary image to combine with line image
    color_edges = np.dstack((line_image, line_image, line_image)) 

    logging.debug("Line-image shape %s" %str(line_image.shape))
    

    # Draw the lines on the original image
    lines_edges = weighted_img(line_image, orig_image, 0.8,  1, 0) 
    
    # Draw the lines on the threshold image image
    thresh_edges = weighted_img(line_image, image, 0.8,  1, 0) 

    # cv2.imwrite(img_dir + "/__lane__" + img_suffix + "." +img_format, lines_edges)
    # cv2.imwrite(img_dir + "/__thresh__" + img_suffix + "." +img_format, thresh_edges)
    return poly, lines_edges, thresh_edges


'''
Get the distortion matrix and coefficients given an image directory with chessboard images.
img_dir : IMage directory.
img_prefix : Prefix of the images to glob
img_format : E.g., jpg, png
nx: Number of chessboard points in the x-direction.
ny: Number of chessboard points in the y-direction.

Returns mtx, dist : Distortion matrix and coefficient
'''
def get_undistort_mtx_and_coeffs(img_dir, img_prefix, img_format, nx, ny):

    if (not os.path.isdir(img_dir)):
        logging.error("Unable to access images dir (%s). Require a valid image dir for camera calibration." %img_dir)
        exit(1)
    
    objp = np.zeros(( nx * ny,3), np.float32)
    objp[:,:2] = np.mgrid[0:nx, 0:ny].T.reshape(-1,2)

    image_paths = glob.glob(img_dir + "/" + img_prefix + "*" + img_format)

    objpoints = []
    imgpoints = []

    for idx, image_path in enumerate(image_paths):
        img = cv2.imread(image_path)
        img_suffix = image_path[image_path.find(img_prefix) + len(img_prefix):len(image_path)]
        img_size = (img.shape[0], img.shape[1])
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Find the chessboard corners
        ret, corners = cv2.findChessboardCorners(gray, (nx,ny), None)

        # If found, add object points, image points
        if ret == True:
            objpoints.append(objp)
            imgpoints.append(corners)

            # Draw and display the corners
            cv2.drawChessboardCorners(img, (8,6), corners, ret)
            write_name = img_dir + '/corners_' + img_suffix
            cv2.imwrite(write_name, img)
        else:
            logging.warning ("Unable to find CB corners for image %s. Skipping" %image_path)
            continue

    logging.debug("OBJP: %s" %str(objpoints))
    # print ("IMGP: %s" %str(imgpoints))

    # Do camera calibration given object points and image points
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, img_size,None,None)

    # Now undistort the images. 
    for idx, image_path in enumerate(image_paths):
        img = cv2.imread(image_path)
        img_suffix = image_path[image_path.find(img_prefix) + len(img_prefix):len(image_path)]
        dst = cv2.undistort(img, mtx, dist, None, mtx)
        write_name = img_dir + '/undist_' + img_suffix
        cv2.imwrite(write_name, dst)

    return mtx, dist

'''
Apply Solel's transform with a minimum of x and y gradients for an input image.


'''
def abs_sobel_threshold(img, min_thresh, max_thresh, orientation='x'):
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # Apply x or y gradient with the OpenCV Sobel() function
    # and take the absolute value
    if orientation == 'x':
        abs_sobel = np.absolute(cv2.Sobel(gray, cv2.CV_64F, 1, 0))
    if orientation == 'y':
        abs_sobel = np.absolute(cv2.Sobel(gray, cv2.CV_64F, 0, 1))
    # Rescale back to 8 bit integer
    scaled_sobel = np.uint8(255*abs_sobel/np.max(abs_sobel))
    # Create a copy and apply the threshold
    binary_output = np.zeros_like(scaled_sobel)
    # Here I'm using inclusive (>=, <=) thresholds, but exclusive is ok too
    binary_output[(scaled_sobel >= min_thresh) & (scaled_sobel <= max_thresh)] = 1

    # Return the result
    return binary_output

# Define a function to return the magnitude of the gradient
# for a given sobel kernel size and threshold values
def mag_threshold(img, sobel_kernel=3, mag_thresh=(0, 255), single_channel=False):
    # Convert to grayscale
    gray = img
    if (not single_channel):
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    # Take both Sobel x and y gradients
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    # Calculate the gradient magnitude
    gradmag = np.sqrt(sobelx**2 + sobely**2)
    # Rescale to 8 bit
    scale_factor = np.max(gradmag)/255 
    gradmag = (gradmag/scale_factor).astype(np.uint8) 
    # Create a binary image of ones where threshold is met, zeros otherwise
    binary_output = np.zeros_like(gradmag)
    binary_output[(gradmag >= mag_thresh[0]) & (gradmag <= mag_thresh[1])] = 1

    # Return the binary image
    return binary_output


# Define a function to threshold an image for a given range and Sobel kernel
def dir_threshold(img, sobel_kernel=3, thresh=(0, np.pi/2)):
    # Grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # Calculate the x and y gradients
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    # Take the absolute value of the gradient direction, 
    # apply a threshold, and create a binary image result
    absgraddir = np.arctan2(np.absolute(sobely), np.absolute(sobelx))
    binary_output =  np.zeros_like(absgraddir)
    binary_output[(absgraddir >= thresh[0]) & (absgraddir <= thresh[1])] = 1

    # Return the binary image
    return binary_output


def process_image(img):
        global img_index, image_format, img_dir, objp
        img = cv2.undistort(img, mtx, dist, None, mtx)
        # img_suffix = image_path[image_path.find(img_prefix) + len(img_prefix):len(image_path)]
        img_suffix =  "_" + str(img_index) + "." + image_format
        imgp = np.zeros((4,2), np.float32)

        ## 
        
        r_channel_img = img[:,:,0]
        g_channel_img = img[:,:,1]
        hls_img = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
        h_channel_img = hls_img[:,:,0]
        l_channel_img = hls_img[:,:,1]
        s_channel_img = hls_img[:,:,2]
        # s_channel_img = img[:,:,2]
        s_thresh = (40, 200)
        r_thresh = (120, 255)
        binary_s = np.zeros_like(s_channel_img)
        binary_s[(s_channel_img > s_thresh[0]) & (s_channel_img <= s_thresh[1])] = 255
        write_name = img_dir + '/s_thresh_' + img_suffix
        logging.debug("New image: %s" %write_name)
        #cv2.imwrite(write_name, binary_s)


        write_name = img_dir + '/h_channel_img' + img_suffix
        logging.debug("New image: %s" %write_name)
        #cv2.imwrite(write_name, h_channel_img)

        write_name = img_dir + '/l_channel_img' + img_suffix
        logging.debug("New image: %s" %write_name)
        #cv2.imwrite(write_name, l_channel_img)

        write_name = img_dir + '/s_channel_img' + img_suffix
        logging.debug("New image: %s" %write_name)
        #cv2.imwrite(write_name, s_channel_img)



        binary_r = np.zeros_like(s_channel_img)
        binary_r[(r_channel_img > r_thresh[0]) & (r_channel_img <= r_thresh[1])] = 255
        write_name = img_dir + '/red_thresh_' + img_suffix
        logging.debug("New image: %s" %write_name)
        # cv2.imwrite(write_name, binary_r)

        binary_rg = np.zeros_like(s_channel_img)
        binary_rg[(r_channel_img > r_thresh[0]) & (r_channel_img <= r_thresh[1])
                  & (g_channel_img > r_thresh[0]) & (g_channel_img <= r_thresh[1])] = 255
        write_name = img_dir + '/rg_thresh_' + img_suffix
        logging.debug("New image: %s" %write_name)
        #cv2.imwrite(write_name, binary_rg)


        color_edges = np.dstack((s_channel_img, s_channel_img, s_channel_img)) 
        mag_thresh_s = mag_threshold(color_edges, sobel_kernel=11, mag_thresh=(40,100)) * 255
        write_name = img_dir + '/mag_thresh_' + img_suffix
        logging.debug("New image: %s" %write_name)
        #cv2.imwrite(write_name, mag_thresh_s)

        
        color_edges = np.dstack((mag_thresh_s, mag_thresh_s, mag_thresh_s)) 
        ret_poly, orig_ret_img, ret_img  = draw_lines_on_image(img, color_edges, img_dir, img_suffix, image_format)
        
        imgp[:,:2] = np.array([ret_poly]).flatten().reshape(4, 2)
        logging.debug ("Imgp: %s" %str(imgp))
        logging.debug ("imgp[0,1]: %s" %str(imgp[2,0]))
        logging.debug ("imgp[2,1] - imgp[0,1] + 1: %s" %str(imgp[1,0] - imgp[2,0] - 1))
        logging.debug ("imgp[2,0]: %s" %str(imgp[1,1]))
        logging.debug ("imgp[3,0]: %s" %str(imgp[3,1]))
        logging.debug ("Size of ret-img:  %s" %str(np.shape(ret_img)))
        # ret_img = ret_img[int(imgp[0,0]): int(imgp[2,0]), int(imgp[2,1]): int(imgp[3,1]), :]
        ret_img = ret_img[int(imgp[0,1]): int(imgp[2,1]), :, :]
        imgp[:,1] = imgp[:,1] - imgp[0,1]
        mat_pers = cv2.getPerspectiveTransform(imgp, objp)
        affine_img = cv2.warpPerspective(ret_img, mat_pers, (250,250))
        
        write_name = img_dir + '/affine_' + img_suffix
        #cv2.imwrite(write_name, affine_img)
        logging.debug ("Size of ret-img-cropped:  %s" %str(np.shape(ret_img)))
        img_size = (int(ret_img.shape[0]), int(ret_img.shape[1]))
        imgpoints.append(imgp)
        objpoints.append(objp)
        img_index = img_index + 1


        # imgs.append(ret_img)
        return orig_ret_img
        # return ret_img
        


def draw_lines_on_images(img_d, img_pr, img_fmt):

    global mtx, dist, image_prefix, image_format, objp, objpoints, imgpoints, use_history

    img_dir = img_d
    if (not os.path.isdir(img_dir)):
        logging.error ("Unable to access images dir (%s). Require a valid image dir for camera calibration." %img_dir)
        exit(1)
    
    
    image_paths = glob.glob(img_dir + "/" + img_pr + "*" + img_fmt)

    
    objp[:,:2] = np.mgrid[50:201:150, 50:201:150].T.reshape(-1,2)
    


    img_size = (2,2)


    mtx, dist = get_undistort_mtx_and_coeffs(img_dir="camera_cal", img_prefix="calibration", img_format="jpg", nx=9, ny=6)  

    image_prefix = img_pr
    image_format = img_fmt

    input_clip = VideoFileClip("project_video.mp4")
    use_history = True
    # reset_history()
    lane_clip = input_clip.fl_image(process_image) #NOTE: this function expects color images!!
    lane_clip.write_videofile("project_res_video.mp4", audio=False)
    use_history = False
    
    '''

    for idx, image_path in enumerate(image_paths):
        img = cv2.imread(image_path)
        imgp = process_image(img)
    
    logging.debug ("Img-points: %s" %str(imgpoints))
    logging.debug ("Obj-points: %s" %str(objpoints))
    '''

    


def main():
    args = parser.parse_args()
    draw_lines_on_images(args.test_img_dir, img_pr="tryst", img_fmt="jpg")
    return 0

    
    

if __name__ == "__main__":
    main()
