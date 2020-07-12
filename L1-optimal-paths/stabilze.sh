#!/bin/bash
for shaky in /home/ishank/Desktop/Regular/*.avi; do
	# Checks if a file like instance
	# actually exists or is just a glob	
	[ -e "$instance" ] || continue
	# Plot arm functions
	echo -e $"Visualizing instance functions $instance\n"
	python3 instance_plotter.py -i "$instance"
	# Slow step so check if output file already exists
	out_name="results/$(basename "$instance" .txt)-out-new.txt"
	# echo $out_name	
	if [ -f "$out_name" ]; then
		echo -e $"Results for instance $instance exist, ... skipping simulation\n"
	else
		echo -e $"Currently Simulating Policies on $instance"
		python3 simulate_policies.py -i "$instance" > "$out_name"
	fi	
	echo -e $"Plotting Simulation results for $instance \n"
	python3 regret_plotter.py -i "$out_name"
done
