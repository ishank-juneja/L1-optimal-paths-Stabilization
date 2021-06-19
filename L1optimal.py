import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
from L1optimal_lpp import stabilize
import argparse
import os
from os.path import join


fourcc_avi = cv.VideoWriter_fourcc(*'XVID')
fourcc_mp4 = cv.VideoWriter_fourcc(*'mp4v')

# Takes im_shape, a tuple and crop ratio, a float < 1.0
def get_corners(im_shape, crop_ratio):
    # Get center of original frames
    img_ctr_x = round(im_shape[1] / 2)
    img_ctr_y = round(im_shape[0] / 2)
    # Get the dimensions w and h of the crop window
    # Crop ratio is a float < 1.0 since the crop window
    # needs to be smaller than the raw frames
    crop_w = round(im_shape[1] * crop_ratio)
    crop_h = round(im_shape[0] * crop_ratio)
    # Get upper left corner of centered crop window
    crop_x = round(img_ctr_x - crop_w / 2)
    crop_y = round(img_ctr_y - crop_h / 2)
    # Return corner points of crop window
    return crop_x, crop_w + crop_x, crop_y, crop_h + crop_y


# Function to plot the original and the corrected/stabilized
# x and y components of the camera trajectory
def plot_trajectory(og, stb, out_path_plot):
    # Print the trajectory and subsequently plot them
    # print("Original camera trajectory (x, y)")
    # print(og)
    # print("--------------------------------")
    # print("Stabilized camera trajectory (x, y)")
    # print(stb)
    # x-coordinate trajectory
    plt.figure()
    plt.plot(og[:, 0])
    plt.plot(stb[:, 0])
    plt.xlabel('Frame Number')
    plt.ylabel('2D Camera x coord. (pixels)')
    plt.title('Original vs Stab x')
    plt.legend(['Original', 'Stabilized'])
    plt.savefig(out_path_plot + "_traj_x.png")
    plt.close()
    # y-coordinate trajectory
    plt.figure()
    plt.plot(og[:, 1])
    plt.plot(stb[:, 1])
    plt.xlabel('Frame Number')
    plt.ylabel('2D Camera y coord. (pixels)')
    plt.title('Original vs Stab y')
    plt.legend(['Original', 'Stabilized'])
    plt.savefig(out_path_plot + "_traj_y.png")
    plt.close()
    return


# Find the inter-frame transformations array F_transforms
# updates F_transforms array inplace
def get_inter_frame_transforms(cap, F_transforms, prev_gray):
    n_frames = F_transforms.shape[0]
    for i in range(n_frames):
        # Detect feature points in previous frame (or 1st frame in 1st iteration)
        prev_pts = cv.goodFeaturesToTrack(prev_gray, maxCorners=200, qualityLevel=0.01,
                                          minDistance=30, blockSize=3)
        # Read next frame
        success, curr = cap.read()
        # If there is no next frame, stream/video has ended
        if not success:
            break
        # Convert to grayscale
        curr_gray = cv.cvtColor(curr, cv.COLOR_BGR2GRAY)
        # Calculate optical flow (i.e. track feature points)
        curr_pts, status, err = cv.calcOpticalFlowPyrLK(prev_gray, curr_gray, prev_pts, None)
        # Sanity check, we should at least get the status of all the features from the previous frame
        # Even if they are no longer being tracked (as indicated by the status array)
        assert prev_pts.shape == curr_pts.shape
        # Filter out and use only valid points
        idx = np.where(status == 1)[0]
        # Update which points we should continue to maintain state for
        prev_pts = prev_pts[idx]
        curr_pts = curr_pts[idx]
        # Find transformation matrix for full 6 DOF affine transform
        m, _ = cv.estimateAffine2D(curr_pts, prev_pts)  # will only work with OpenCV-3 or less
        # print(m.shape) >> (2, 3) since 6 DOF full affine transform
        # Add current transformation matrix $F_t$ to array
        # $F_t$ is a right multiplied homogeneous affine transform, last column is untouched
        # We start from index 1 since index 0 is the identity matrix by construction/definition
        F_transforms[i + 1, :, :2] = m.T
        # Move to next frame
        prev_gray = curr_gray
        # print("Frame: " + str(i) + "/" + str(n_frames) + " -  Tracked points : " + str(len(prev_pts)))
    return


def write_output(cap, out, B_transforms, shape, crop_ratio):
    # Reset stream to first frame
    cap.set(cv.CAP_PROP_POS_FRAMES, 0)
    n_frames = B_transforms.shape[0]
    # print("Inside write_output, number of frames are {0}".format(n_frames))
    # Write n_frames transformed frames
    for i in range(n_frames):
        # Read the first/next frame
        success, frame = cap.read()
        # cv.imshow("Before and After", frame)
        # cv.waitKey(10)
        # If there is not next frame to read, exit display loop
        if not success:
            break
        # Apply affine wrapping to the given frame
        # Also convert to sta
        scale_x = 1 / crop_ratio
        scale_y = 1 / crop_ratio

        scaling_matrix = np.eye(3, dtype=float)
        scaling_matrix[0][0] = scale_x
        scaling_matrix[1][1] = scale_y

        shifting_to_center_matrix = np.eye(3, dtype=float)
        shifting_to_center_matrix[0][2] = -shape[0] / 2.0
        shifting_to_center_matrix[1][2] = -shape[1] / 2.0

        shifting_back_matrix = np.eye(3, dtype=float)
        shifting_back_matrix[0][2] = shape[0] / 2.0
        shifting_back_matrix[1][2] = shape[1] / 2.0

        B_matrix = np.eye(3, dtype=float)
        B_matrix[:2][:] = B_transforms[i, :, :2].T
        final_matrix = shifting_back_matrix @ scaling_matrix @ shifting_to_center_matrix @ np.linalg.inv(B_matrix)
        frame_stabilized = cv.warpAffine(frame, final_matrix[:2, :], shape)
        # frame_stabilized = cv.warpAffine(frame, B_transforms[i, :, :2].T, shape)
        # Write the frame to the file
        # frame_out = cv.hconcat([frame, frame_stabilized])
        frame_out = frame_stabilized
        # If the image is too big, resize it.
        # if frame_out.shape[1] > 1920:
        # frame_out = cv.resize(frame_out, (frame_out.shape[1], frame_out.shape[0]))
        # Display the result in a window before writing it to file
        # cv.imshow("Before and After", frame_out)
        # cv.waitKey(10)
        out.write(frame_out)
    out.release()
    return


# Main function reads in the input, processes it and writes the output
def main(args, in_file_path, out_file_path, fourcc):
    file = in_file_path
    # Extract input file name sans extension
    in_name = file.split('/')[-1].split('.')[0]
    # crop_ratio = 0.7
    crop_ratio = args.crop_ratio
    # Read input video
    cap = cv.VideoCapture(file)
    # Get frame count, possible apriori if reading a file
    n_frames = int(cap.get(cv.CAP_PROP_FRAME_COUNT))
    print("Number of frames in file are {0}".format(n_frames))
    # Get width and height of frames in video stream from cap object
    w = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
    # Define the codec for output video
    # fourcc = cv.VideoWriter_fourcc(*'MPEG')
    # Get input fps, use same for output
    fps = int(cap.get(cv.CAP_PROP_FPS))
    # Pre-define transformation-store array
    # Uses 3 parameters since it is purely a coordinate transform
    # A collection of n_frames number of  homography matrices
    F_transforms = np.zeros((n_frames, 3, 3), np.float32)
    # Initialise all transformations with Identity matrix
    F_transforms[:, :, :] = np.eye(3)
    # Read first frame
    _, prev = cap.read()
    # Convert frame to grayscale for feature tracking using openCV
    prev_gray = cv.cvtColor(prev, cv.COLOR_BGR2GRAY)
    # Find the inter-frame transformations array F_transforms
    get_inter_frame_transforms(cap, F_transforms, prev_gray)
    # Get stabilization transforms B_t by processing motion transition transforms F_t
    B_transforms = stabilize(F_transforms, prev.shape, True, None, crop_ratio)
    # Accumulate by right multiplication into C_trajectory
    # $C_{t + 1} = C_t x F_t$
    # Initialise camera trajectory as a copy since it would help with stealing index 0 and the shape of F itself
    C_trajectory = F_transforms.copy()
    for i in range(1, n_frames):
        # Right multiply transformations to accumulate all the changes into camera trajectory
        C_trajectory[i, :, :] = C_trajectory[i - 1, :, :] @ F_transforms[i, :, :]
    # Repeat right multiplication procedure to obtain stabilized camera trajectory P
    P_trajectory = C_trajectory.copy()
    # Apply transform to C_trajectory to get P_trajectory
    for i in range(n_frames):
        P_trajectory[i, :, :] = C_trajectory[i, :, :] @ B_transforms[i, :, :]
    # if the plotting camera trajectory flag is passed, plot trajectories
    if args.trajPlot:
        # Starting coordinate (0, 0) in homogeneous system
        origin = np.array([0, 0, 1])
        # Evolution of coordinate of camera trajectory under original scheme
        evolution_og = origin @ C_trajectory
        # Evolution of origin under stabilized trajectory
        evolution_stab = origin @ P_trajectory
        # Get output path
        out_path_camera_plot = join(os.path.dirname(out_file_path), 'plots', in_name)
        plot_trajectory(evolution_og, evolution_stab, out_path_camera_plot)
    # Object to write the output video
    out = cv.VideoWriter(out_file_path, fourcc, fps, (w, h))
    # Stabilize the video frame by frame using the obtained transforms and save it
    write_output(cap, out, B_transforms, (w, h), crop_ratio)
    cap.release()
    return


if __name__ == "__main__":
    # Pass command line inputs for the stabilization procedure
    parser = argparse.ArgumentParser()
    # Add input file path, default type is string
    parser.add_argument("-i", action="store", dest="file")
    # Add output file path, default type is string
    parser.add_argument("-o", action="store", dest="file_out")
    # Crop ratio to avoid black corners creeping into the main stabilized video frame area
    parser.add_argument("-crop-ratio", action="store", dest="crop_ratio", type=float)
    # Boolean argument that is True if the flag --trajPlot is passed
    parser.add_argument("--trajPlot", action="store_true")
    # read cmd line arguments
    args_read = parser.parse_args()
    # in file path
    in_file = args_read.file
    out_path = args_read.file_out
    # Extract input file name sans extension and vid type of either mp4 or avi
    [in_name, vid_type] = in_file.split('/')[-1].split('.')
    # Define the codec for output video
    if vid_type == 'mp4':
        fourcc = fourcc_mp4
    elif vid_type == 'avi':
        fourcc = fourcc_avi
    else:
        print("Unsupported video file type")
        exit(-1)
    main(args_read, in_file, out_path, fourcc)
