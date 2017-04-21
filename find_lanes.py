import argparse
import base64
from datetime import datetime
import os
import shutil
import copy

import numpy as np

import cv2
import glob

parser = argparse.ArgumentParser(description='Advanced lane finding project.')
parser.add_argument('-c', '--cb_img_dir', default="./camera_cal", help='Chess-board images dir.')
parser.add_argument('-nx', '--nx', default=9, help='Number of chessboard corners in the x-dimension')
parser.add_argument('-ny', '--ny', default=6, help='Number of chessboard corners in the y-dimension')
parser.add_argument('-f', '--img_format', default="jpg", help='Format of camera calibration images')
parser.add_argument('-p', '--img_prefix', default="calibration", help='Prefix for calibration images (used in glob)')

args = parser.parse_args()

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
        print ("Unable to access images dir (%s). Require a valid image dir for camera calibration." %args.cb_img_dir)
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
        ret, corners = cv2.findChessboardCorners(gray, (args.nx,args.ny), None)

        # If found, add object points, image points
        if ret == True:
            objpoints.append(objp)
            imgpoints.append(corners)

            # Draw and display the corners
            cv2.drawChessboardCorners(img, (8,6), corners, ret)
            write_name = img_dir + '/corners_' + img_suffix
            cv2.imwrite(write_name, img)
        else:
            print ("Unable to find CB corners for image %s. Skipping" %image_path)
            continue

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

mtx, dist = get_undistort_mtx_and_coeffs(args.cb_img_dir, args.img_prefix, args.img_format, args.nx, args.ny)

