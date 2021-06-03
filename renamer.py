import os
from tqdm import tqdm
import time
import sys
from os import path

counter = 1
name = sys.argv[0]
DIR = os.path.realpath(__file__).replace(__file__,"")

files = os.listdir(DIR+'DATASETS\\')

for file in tqdm(files):
	# print(file)
	ls_dir = 'DATASETS\\'+file
	if(path.isfile(ls_dir)):pass
	else:
		counter
		counter = 0
		for x in os.listdir(DIR+ls_dir):
			if(path.isfile(ls_dir+"\\"+x)):
				new_name = "{}.jpg".format(counter)
				os.system("rename {} {}".format(DIR+ls_dir+"\\"+x,new_name))
				counter +=1
		os.rename(ls_dir,(ls_dir.replace(" ","_").replace(",","")).lower())
print(DIR)