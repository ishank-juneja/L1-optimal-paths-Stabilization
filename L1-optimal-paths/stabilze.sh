#!/bin/bash
for instance in /home/ishank/Desktop/Regular/*.avi; do
	# Checks if a file like instance
	# actually exists or is just a glob	
	[ -e "$instance" ] || continue
	# Plot arm functions
	echo -e $"Stabilizing file $instance\n"
	python3 stabilization_L1_optimal.py -i "$instance"
done
