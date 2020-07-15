# Averaging Based Video Stabilization

Code borrowed from this [tutorial](https://www.learnopencv.com/video-stabilization-using-point-feature-matching-in-opencv/)  

Stabilizes a shaky video, in the offline mode, by 

1. Obtaining the Shaky camera trajectory for the entire frame sequence
2. Smoothening this camera trajectory using an averaging (or otherwise any low pass) filter
3. Use the stabilized camera trajectory to construct a stabilized video sequence by applying warping transforms since transforms in the homogeneous coordinate system can be composed together

