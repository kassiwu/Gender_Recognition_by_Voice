#!/bin/bash
# EE695 Final Project Data Manager
#Show location of the script
DIR="$( cd "$( dirname "$0" )" && pwd )"	#current location
#echo "Script location: ${DIR}"

Data=~/Desjtop/Final-Project/Data/Original/*	# All data folder
ARCH_loc=~/Desjtop/Final-Project/Data/Processed		# Archive location

M_wav=~/Desjtop/Final-Project/Data/Male	# folder for processed MALE speech samples
F_wav=~/Desjtop/Final-Project/Data/Female	# foleder for processed FEMALE speech samples

M_loc=~/Desjtop/Final-Project/Data/Male/*
F_loc=~/Desjtop/Final-Project/Data/Female/*



counter_folder=0	# count processed data folders 

CF=0
CM=0
# Count processed samples exist in the target folder 
for wav_m in $M_loc;do
	let CM+=1
done

for wav_f in $F_loc;do
	let CF+=1
done

for folder in $Data; do
	gender_flag=-1	#initial genderflag 

	if [ -d $folder ]; then #Data folder exists
	
		label_file=$folder/etc/README	#file with speak's information 
		wav_folder=$folder/wav/*	#sound track folder

		# Read label file to obtain the gender of the speaker
		if [ -f $label_file ]; then	#if if file exists
			IFS=''
			while  read -r line; do	#read the file by lines
				if [ "$line" == "Gender: Male" ]; then 
					let CM+=1
					gender_flag=0
					break
				elif [ "$line" == "Gender: Female" ]; then
					let CF+=1
					gender_flag=1
					break
				fi
			done < $label_file		#Close file 
		fi
		
		#Move .wav file to target foleder acording to the gender of the speakers
		case $gender_flag in		
		0)	# Male Speaker 
			echo "Male"
			for Sample in $wav_folder; do
				if [ ${Sample: -4} == ".wav" ]; then
					cp $Sample $M_wav/M$CM.wav
					break
				else
					continue
				fi
			# done 
			mv $folder $ARCH_loc		# move processed folder to archive location 
			let counter_folder+=1		#count processed folder 
			;;
		1)	# Female Speaker
			echo "Female"
			for Sample in $wav_folder; do
				if [ ${Sample: -4} == ".wav" ]; then
					cp $Sample $F_wav/F$CF.wav
					break
				else	
					continue
				fi
			done 
			mv $folder $ARCH_loc		# move processed folder to archive location  
			let counter_folder+=1		#count processed folder
			;;
		*)	# Unable to identify speaker from label file 
			echo "Error in label file $label_file."
			;;
		esac
	fi
done

echo "number of foulder processed: $counter_folder"
echo "number of male samples: $CM"
echo "number of female samples: $CF"

