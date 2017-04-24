
# coding: utf-8

# # **Finding Lane Lines on the Road** 
# ***
# In this project, you will use the tools you learned about in the lesson to identify lane lines on the road.  You can develop your pipeline on a series of individual images, and later apply the result to a video stream (really just a series of images). Check out the video clip "raw-lines-example.mp4" (also contained in this repository) to see what the output should look like after using the helper functions below. 
# 
# Once you have a result that looks roughly like "raw-lines-example.mp4", you'll need to get creative and try to average and/or extrapolate the line segments you've detected to map out the full extent of the lane lines.  You can see an example of the result you're going for in the video "P1_example.mp4".  Ultimately, you would like to draw just one line for the left side of the lane, and one for the right.
# 
# ---
# Let's have a look at our first image called 'test_images/solidWhiteRight.jpg'.  Run the 2 cells below (hit Shift-Enter or the "play" button above) to display the image.
# 
# **Note: If, at any point, you encounter frozen display windows or other confounding issues, you can always start again with a clean slate by going to the "Kernel" menu above and selecting "Restart & Clear Output".**
# 
# ---

# **The tools you have are color selection, region of interest selection, grayscaling, Gaussian smoothing, Canny Edge Detection and Hough Tranform line detection.  You  are also free to explore and try other techniques that were not presented in the lesson.  Your goal is piece together a pipeline to detect the line segments in the image, then average/extrapolate them and draw them onto the image for display (as below).  Once you have a working pipeline, try it out on the video stream below.**
# 
# ---
# 
# <figure>
#  <img src="line-segments-example.jpg" width="380" alt="Combined Image" />
#  <figcaption>
#  <p></p> 
#  <p style="text-align: center;"> Your output should look something like this (above) after detecting line segments using the helper functions below </p> 
#  </figcaption>
# </figure>
#  <p></p> 
# <figure>
#  <img src="laneLines_thirdPass.jpg" width="380" alt="Combined Image" />
#  <figcaption>
#  <p></p> 
#  <p style="text-align: center;"> Your goal is to connect/average/extrapolate line segments to get output like this</p> 
#  </figcaption>
# </figure>

# **Run the cell below to import some packages.  If you get an `import error` for a package you've already installed, try changing your kernel (select the Kernel menu above --> Change Kernel).  Still have problems?  Try relaunching Jupyter Notebook from the terminal prompt.  Also, see [this forum post](https://carnd-forums.udacity.com/cq/viewquestion.action?spaceKey=CAR&id=29496372&questionTitle=finding-lanes---import-cv2-fails-even-though-python-in-the-terminal-window-has-no-problem-with-import-cv2) for more troubleshooting tips.**  

# In[1]:

#importing some useful packages
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2
get_ipython().magic('matplotlib inline')


# In[2]:

#reading in an image
image = mpimg.imread('test_images/solidWhiteRight.jpg')
#printing out some stats and plotting
print('This image is:', type(image), 'with dimesions:', image.shape)
plt.imshow(image)  # if you wanted to show a single color channel image called 'gray', for example, call as plt.imshow(gray, cmap='gray')


# **Some OpenCV functions (beyond those introduced in the lesson) that might be useful for this project are:**
# 
# `cv2.inRange()` for color selection  
# `cv2.fillPoly()` for regions selection  
# `cv2.line()` to draw lines on an image given endpoints  
# `cv2.addWeighted()` to coadd / overlay two images
# `cv2.cvtColor()` to grayscale or change color
# `cv2.imwrite()` to output images to file  
# `cv2.bitwise_and()` to apply a mask to an image
# 
# **Check out the OpenCV documentation to learn about these and discover even more awesome functionality!**

# Below are some helper functions to help get you started. They should look familiar from the lesson!

# In[3]:

import math
from scipy.stats import linregress

# Global variables used for averaging over images.
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


def set_history_len(hl):
    '''
    Set the history length and determine weights 
    '''
    global hist_length
    global weights
    
    hist_length = hl
    cur_wt = 1.0
    for i in xrange(hl):
       weights[i] = cur_wt / 2.0 
    
def grayscale(img):
    """Applies the Grayscale transform
    This will return an image with only one color channel
    but NOTE: to see the returned image as grayscale
    (assuming your grayscaled image is called 'gray')
    you should call plt.imshow(gray, cmap='gray')"""
    return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # Or use BGR2GRAY if you read an image with cv2.imread()
    # return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
def canny(img, low_threshold, high_threshold):
    """Applies the Canny transform"""
    return cv2.Canny(img, low_threshold, high_threshold)

def gaussian_blur(img, kernel_size):
    """Applies a Gaussian Noise kernel"""
    return cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)

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
    cv2.fillPoly(mask, vertices, ignore_mask_color)
    
    #returning the image only where mask pixels are nonzero
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image

def length(line):
    '''
    Get the length of a line. 
    NOTE: this is not used. See my notes. Used in one of the intermediate version of my algorithm.
    '''
    x1, y1, x2, y2 = [float(a) for a in line]
    return math.sqrt((x2-x1)*(x2-x1) + (y2-y1)*(y2-y1))

def weights(in_list):
    '''
    Get the weighted coordinate of a line. 
    NOTE: this is not used. See my notes. Used in one of the intermediate version of my algorithm.
    '''
    total_wt = sum(in_list)
    if (total_wt == 0):
        return in_list
    return [(w / total_wt) for w in in_list]


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

def int_mid(coords):
    '''
    Get the median of  a set of points.
    '''
    med = np.median(coords)
    if not math.isnan(med):
        return int(med)
    else:
        return 0

    
def int_avg(coords):
    '''
    Get the mean of a set of points.
    NOTE: this is not used. See my notes. Used in the first version of my algorithm.
    '''
    avg = np.average(coords)
    if not math.isnan(avg):
        return int(avg)
    else:
        return 0

def prune_lines(bottom, top, weights, side, width):
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
            if (DEBUG):
                print ("Removing noisy line: side=%d, width=%d (%d, %d)" %(side, width, t, b))
            top.pop(index)
            bottom.pop(index)
            weights.pop(index)
        else:
            index += 1

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

#def reject_outliers(top, bottom, m=1.5):
#    top_mean = np.median(data)
#    for t in top:
        
#    return data[abs(data - np.mean(data)) < m * np.std(data)]


def draw_lines(img, lines, color=[255, 0, 0], thickness=2):
    """
    NOTE: this is the function you might want to use as a starting point once you want to 
    average/extrapolate the line segments you detect to map out the full
    extent of the lane (going from the result shown in raw-lines-example.mp4
    to that shown in P1_example.mp4).  
    
    Think about things like separating line segments by their 
    slope ((y2-y1)/(x2-x1)) to decide which segments are part of the left
    line vs. the right line.  Then, you can average the position of each of 
    the lines and extrapolate to the top and bottom of the lane.
    
    This function draws `lines` with `color` and `thickness`.    
    Lines are drawn on the image inplace (mutates the image).
    If you want to make the lines semi-transparent, think about combining
    this function with the weighted_img() function below
    """
    # First sort the lines into left and right.
    img_center_x = img.shape[1] / 2;
    
    # A line is (x1, y1, x2, y2). 
    # We're just checking if x1 falls to left of image center.
    left_lines = [line[0] for line in lines if (line[0][0] < img_center_x)]#get_left_lines(lines, img_center_x)
    left_line_weights = weights([length(line) for line in left_lines])
    
    
    # A line is (x1, y1, x2, y2). 
    # We're just checking if x1 falls to right of image center.
    right_lines = [line[0] for line in lines if (not line[0][0] < img_center_x)] #get_right_lines(lines, img_center_x)
    right_line_weights = weights([length(line) for line in right_lines])
    
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
    
    #left_bottom_x = int ((weighted_avg(left_bottom_xs, left_line_weights) 
    #                + int_mid(left_bottom_xs) ) / 2)
    #left_top_x = int((weighted_avg(left_top_xs, left_line_weights)  
    #                 + int_mid(left_top_xs)) / 2)
    
    #right_bottom_x = int ((weighted_avg(right_bottom_xs, right_line_weights) 
    #                       + int_mid(right_bottom_xs)) / 2)
    #right_top_x = int ((weighted_avg(right_top_xs, right_line_weights) 
    #                    + int_mid(right_top_xs) ) / 2)
   
    global use_history, weights, left_bottom_hist, left_top_hist, right_bottom_hist, right_top_hist

    if (use_history):
        # Sanity check. If sanity check fails, and history is enabled, just use the previous frame's value.
        if (left_top_x > right_top_x or left_bottom_x > right_bottom_x 
           or left_top_x < left_bottom_x or right_top_x > right_bottom_x):
            # Sanity check failed. Use previous frame's values  available
            if (DEBUG):
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
    cv2.line(img, (left_bottom_x, img.shape[0]), (left_top_x, int(img.shape[0] * 0.62)), color, 8)
    cv2.line(img, (right_bottom_x, img.shape[0]), (right_top_x, int(img.shape[0] * 0.62)), color, 8)
    

def hough_lines(img, rho, theta, threshold, min_line_len, max_line_gap):
    """
    `img` should be the output of a Canny transform.
        
    Returns an image with hough lines drawn.
    """
    lines = cv2.HoughLinesP(img, rho, theta, threshold, np.array([]), minLineLength=min_line_len, maxLineGap=max_line_gap)
    line_img = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
    draw_lines(line_img, lines)
    return line_img

# Python 3 has support for cool math symbols.

def weighted_img(img, initial_img, α=0.8, β=1., λ=0.):
    """
    `img` is the output of the hough_lines(), An image with lines drawn on it.
    Should be a blank image (all black) with lines drawn on it.
    
    `initial_img` should be the image before any processing.
    
    The result image is computed as follows:
    
    initial_img * α + img * β + λ
    NOTE: initial_img and img must be the same shape!
    """
    return cv2.addWeighted(initial_img, α, img, β, λ)


# ## Test on Images
# 
# Now you should build your pipeline to work on the images in the directory "test_images"  
# **You should make sure your pipeline works well on these images before you try the videos.**

# In[4]:

import os
test_dir = "test_images"
os.listdir(test_dir)
img_index = 0



def draw_lines_on_image(image, name = ""):
    ## Copy-paste from quiz.
    gray = grayscale(image)
    global img_index
    if (name == ""):
        image_file = str(img_index) + ".jpg"
        img_index += 1
    #print ("Gray image: ")
    #plt.imshow(gray)

    ret,bin_thresh = cv2.threshold(gray, 160, 255,cv2.THRESH_BINARY)
    #bin_thresh = cv2.equalizeHist(gray)
    
    # Define a kernel size and apply Gaussian smoothing
    blur_gray = gaussian_blur(bin_thresh, 5)
    
    
    #print ("Smoothed image: ")
    #plt.imshow(blur_gray, cmap='gray')

    # Define our parameters for Canny and apply
    low_threshold = 100
    high_threshold = 150
    edges = canny(blur_gray, low_threshold, high_threshold)
    #print ("Edges: ")
    #plt.imshow(edges, cmap='gray')

    # Next we'll create a masked edges image using cv2.fillPoly()
    mask = np.zeros_like(edges)   
    ignore_mask_color = 255   

    # This time we are defining a four sided polygon to mask
    imshape = image.shape
    vertices = np.array([[(imshape[1] *0.48, imshape[0] * 0.62),(imshape[1] * 0.52, imshape[0] * 0.62),            (imshape[1] * 1. , imshape[0]), (imshape[1] * 0.0, imshape[0])]], dtype=np.int32)
    masked_edges = region_of_interest(edges, vertices)
    
    #masked_edges = cv2.bitwise_and(edges, mask)
    #print ("Masked Edges: ")
    #plt.imshow(masked_edges, cmap='gray')

    # Define the Hough transform parameters
    # Make a blank the same size as our image to draw on
    rho = 1 # distance resolution in pixels of the Hough grid
    theta = np.pi/180 # angular resolution in radians of the Hough grid
    threshold = 20     # minimum number of votes (intersections in Hough grid cell)
    min_line_length = 2 #minimum number of pixels making up a line
    max_line_gap = 350    # maximum gap in pixels between connectable line segments
    line_image = np.copy(image)*0 # creating a blank to draw lines on

    # Run Hough on edge detected image
    # Output "lines" is an array containing endpoints of detected line segments
    line_image = hough_lines(masked_edges, rho, theta, threshold, min_line_length, max_line_gap)
    #print ("line_image ")
    #plt.imshow(line_image, cmap='gray')
    
    # Iterate over the output "lines" and draw lines on a blank image
    #draw_lines(line_image,lines,(255,0,0),10)
    #for line in lines:
    #    for x1,y1,x2,y2 in line:
    #        draw_lines(line_image,(x1,y1),(x2,y2),(255,0,0),10)

    # Create a "color" binary image to combine with line image
    color_edges = np.dstack((line_image, line_image, line_image)) 

    # Draw the lines on the edge image
    lines_edges = weighted_img(line_image, image, 0.8,  1, 0) 
    #cv2.fillPoly(mask, line_image, ignore_mask_color)
    #plt.imshow(lines_edges)
    #print ("line_image ")
    plt.imshow(lines_edges)
    cv2.imwrite(test_dir + "/__lane__" + image_file, lines_edges)
    return lines_edges
    
    

for image_file in os.listdir(test_dir):
    # Output remnant image
    if image_file.startswith("__lane__"):
        continue
    
    image = mpimg.imread(test_dir + "/" + image_file)
    
    draw_lines_on_image(image)
    


# run your solution on all test_images and make copies into the test_images directory).

# In[5]:

# TODO: Build your pipeline that will draw lane lines on the test_images
# then save them to the test_images directory.



# ## Test on Videos
# 
# You know what's cooler than drawing lanes over images? Drawing lanes over video!
# 
# We can test our solution on two provided videos:
# 
# `solidWhiteRight.mp4`
# 
# `solidYellowLeft.mp4`
# 
# **Note: if you get an `import error` when you run the next cell, try changing your kernel (select the Kernel menu above --> Change Kernel).  Still have problems?  Try relaunching Jupyter Notebook from the terminal prompt. Also, check out [this forum post](https://carnd-forums.udacity.com/questions/22677062/answers/22677109) for more troubleshooting tips.**
# 
# **If you get an error that looks like this:**
# ```
# NeedDownloadError: Need ffmpeg exe. 
# You can download it by calling: 
# imageio.plugins.ffmpeg.download()
# ```
# **Follow the instructions in the error message and check out [this forum post](https://carnd-forums.udacity.com/display/CAR/questions/26218840/import-videofileclip-error) for more troubleshooting tips across operating systems.**

# In[9]:

# Import everything needed to edit/save/watch video clips
from moviepy.editor import VideoFileClip
from IPython.display import HTML


# In[10]:

def process_image(image):
    # NOTE: The output you return should be a color image (3 channel) for processing video below
    # TODO: put your pipeline here,
    # you should return the final output (image with lines are drawn on lanes)

    #return result
    return draw_lines_on_image(image)


# Let's try the one with the solid white lane on the right first ...

# In[11]:

white_output = 'white.mp4'
clip1 = VideoFileClip("solidWhiteRight.mp4")
use_history = True
reset_history()
white_clip = clip1.fl_image(process_image) #NOTE: this function expects color images!!
use_history = False
get_ipython().magic('time white_clip.write_videofile(white_output, audio=False)')


# Play the video inline, or if you prefer find the video in your filesystem (should be in the same directory) and play it in your video player of choice.

# In[12]:

HTML("""
<video width="960" height="540" controls>
  <source src="{0}">
</video>
""".format(white_output))


# **At this point, if you were successful you probably have the Hough line segments drawn onto the road, but what about identifying the full extent of the lane and marking it clearly as in the example video (P1_example.mp4)?  Think about defining a line to run the full length of the visible lane based on the line segments you identified with the Hough Transform.  Modify your draw_lines function accordingly and try re-running your pipeline.**

# Now for the one with the solid yellow lane on the left. This one's more tricky!

# In[14]:

yellow_output = 'yellow.mp4'
clip2 = VideoFileClip('solidYellowLeft.mp4')
use_history = True
reset_history()
yellow_clip = clip2.fl_image(process_image)
use_history = False
get_ipython().magic('time yellow_clip.write_videofile(yellow_output, audio=False)')


# In[15]:

HTML("""
<video width="960" height="540" controls>
  <source src="{0}">
</video>
""".format(yellow_output))


# In[19]:

## Reflections

get_ipython().set_next_input('Congratulations on finding the lane lines!  As the final step in this project, we would like you to share your thoughts on your lane finding pipeline... specifically, how could you imagine making your algorithm better / more robust?  Where will your current algorithm be likely to fail');get_ipython().magic('pinfo fail')

Please add your thoughts below,  and if you're up for making your pipeline more robust, be sure to scroll down and check out the optional challenge video below!

---

This was a decent challenge in getting the image processing pipeline ready. As with every new environment, the setup took way more time than one would anticipate. 

I'm also getting used to python3 differences from python2. This is the first time I've used python3. 

I started off with a fairly simplistic algorithm. 

* Divide the frame into left and right halves. 
* Only consider the bottom ~40% of the image for lane markings. 
* Blur the image, find the lines in each of left and right halves and average them intercepts. 

This algorithm did not perform well for the yellow lines. I then tried various weighted average schemes based on the length of the lines. They worked in some cases and they failed badly in the challenge scheme.

The algorithm I'm submitting has some simple changes to the basic algorithm described above.
* I consider the median of the intercepts, rather than averaging. This removes the outliers completely. 
* I threshold the image first, this greatly boosts the detection accuracy in presence of shadows. 
* The median of intercepts seemed to cause more than desired jitter. So, I have a weighted average of the history of the intercepts. This works since the current position of the car, is a good indicator of the future position of the car. 
  * A history of 7 frames seemed to be a good choice. I considered 3 - 7. 
* I've pruned lines that should obviously not be considered. For example, a left side line crossing a valid right side line. 



---


# ## Submission
# 
# If you're satisfied with your video outputs it's time to submit!  Submit this ipython notebook for review.
# 

# ## Optional Challenge
# 
# Try your lane finding pipeline on the video below.  Does it still work?  Can you figure out a way to make it more robust?  If you're up for the challenge, modify your pipeline so it works with this video and submit it along with the rest of your project!

# In[20]:

challenge_output = 'extra.mp4'
clip2 = VideoFileClip('challenge.mp4')
use_history = True
reset_history()
challenge_clip = clip2.fl_image(process_image)
get_ipython().magic('time challenge_clip.write_videofile(challenge_output, audio=False)')


# In[21]:

HTML("""
<video width="960" height="540" controls>
  <source src="{0}">
</video>
""".format(challenge_output))


# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:



