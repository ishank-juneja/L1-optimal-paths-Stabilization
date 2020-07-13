import cv2 as cv


# Methods for affine transforms

# Has the restriction of only 4 free parameters
# So called constrained affine and not full affine transform
# https://docs.opencv.org/4.0.0/d9/d0c/group__calib3d.html#gad767faff73e9cbd8b9d92b955b50062d
cv.estimateAffinePartial2D()

# Also see
# https://stackoverflow.com/questions/56229484/how-to-improve-accuracy-of-estimateaffine2d-or-estimagerigidtransform-in-openc

# For complete 6 free parameters see
cv.estimateAffine2D()
# http://amroamroamro.github.io/mexopencv/matlab/cv.estimateAffine2D.html
# Complete documentation of above function
# Also the below link for another version of this documentation
# https://docs.opencv.org/master/d9/d0c/group__calib3d.html#ga27865b1d26bac9ce91efaee83e94d4dd

# app source code for both these methods can be found here
# https://github.com/opencv/opencv/blob/31ec9b2aa723e36ef1f9280b9ba074e6264f51e6/modules/calib3d/src/ptsetreg.cpp

