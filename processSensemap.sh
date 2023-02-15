#!/bin/bash
# This is a script to run the surface mapping pipeline on an entire
# BIDS structured directory or to process a single BIDS structured
# dataset.
#
# The first input needs to be the path to the BIDS directory. The
# second input is optional in case a single subject is to be
# processed only. In that case use the subject's ID as second
# input, e.g.
#
# 	processSensemap.sh /tmp/luesebrink/sensemap/
#
# will run the Python script 'surfaceMapping.py' on the entire
# specified folder. Whereas,
#
#	processSensemap.sh /tmp/luesebrin/sensemap/ aaa
#
# will run the script for the subject 'aaa' only.
#
# Written by Falk Luesebrink
# falk (dot) luesebrink (at) ovgu (dot) de
#
# Version 1.0
# 22.11.2022
#
path=$1
sub=$2

echo ""
echo "------------------------------------------------------------------------------"
echo "| Running surface mapping script"
echo "------------------------------------------------------------------------------"

if [[ -d ${path} ]] && [[ "$#" -eq 1 ]]; then
	directories=$(find ${path} -maxdepth 1 -type d -name "sub-*" | sort)
	
	for directory in ${directories}; do
		sub=$(echo ${directory} | tail -c 4) 
		if [[ -f ${path}/derivatives/sub-${sub}/processed_successfully ]]; then
			echo "Subject ${sub} already processed. Skipping re-processing it."
			echo ""
		elif [[ -f ${path}/derivatives/sub-${sub}/is_running ]]; then
			echo "*********************** WARNING ***********************"
			echo ""
			echo "Subject ${sub} is being processed currently. Skipping subject."
			echo ""
			echo "*********************** WARNING ***********************"
			echo "In case the subject is not being processing currently,"
			echo "please check log file to identify potential errors. In"
			echo "order to continue processing, please delete"
			echo ""
			echo "${path}/derivatives/sub-${sub}/is_running"
			echo ""
			echo "and re-start the script."
			echo "*********************** WARNING ***********************"
			echo ""
		else
			mkdir -p ${path}/derivatives/sub-${sub}/
			python3 -u layerMapping.py ${sub} $path} |& tee ${path}/derivatives/sub-${sub}/layerMapping_${sub}.log
		fi
	done
elif [[ -d ${path} ]] && [[ "$#" -eq 2 ]]; then
	if [[ -f ${path}/derivatives/sub-${sub}/processed_successfully ]]; then
		echo "Subject ${sub} already processed. Skipping re-processing it."
		echo ""
	elif [[ -f ${path}/derivatives/sub-${sub}/is_running ]]; then
		echo "Subject ${sub} is being processed currently. Skipping subject."
		echo ""
		echo "********************** WARNING **********************"
		echo "In case the subject is not being processing currently. please delete"
		echo ""
		echo "${path}/derivatives/sub-${sub}/is_running"
		echo ""
		echo "and re-start the process."
		echo "********************** WARNING **********************"
		echo ""
	else
    		mkdir -p ${path}/derivatives/sub-${sub}/
        	python3 -u layerMapping.py ${sub} ${path} |& tee ${path}/derivatives/sub-${sub}/layerMapping_${sub}.log
	fi
else
	echo "Something went wrong?"
fi
