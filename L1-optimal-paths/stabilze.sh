#!/bin/bash
for unstable_video in /home/ishank/courses_IITB/unstable_video_data_Regular/*.avi; do
    # Checks if a file like unstable_video
    # actually exists or is just a glob	in itself
    [ -e "$unstable_video" ] || continue
    # Check if it is already a stabilized video containing
    # the keyword 'stb' inside it
    if [[ $unstable_video == *"stb"* ]]; then
        continue
    else
        # Plot arm functions
        echo -e $"--------------------------------------------\n"
        echo -e $"Stabilizing file $unstable_video\n"
        python3 stabilization_L1_optimal.py -i "$unstable_video" -crop-ratio 0.7
    fi
done
