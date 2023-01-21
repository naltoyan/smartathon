import csv
import numpy as np
from PIL import Image
import glob
import os
import cv2
import glob as glob
import pandas as pd

#This function will extract images for the detected VP by yolov5 model and write csv file with their info
def get_detectedVP(txtfilepath,csvFileName,imPath,image_path1)
	#txtfilepath is the path for txt files outputted by yolo, in our test is 'dataset/labels/*.txt'
	#csvFileName is the desired name for the info csv file, in our test is 'dataset/testsetPatches.csv'
	#imPath is the path for the desired folder to save the extracted images, in our test is'dataset/testset/'
	#image_path1 is the path for test images, in our test is "/dataset/images/"
	
	#output csv header
	header=['id','image_path','name','xmax','xmin','ymax','ymin']
	#Names of classes
	category=['GRAFFITI','FADED_SIGNAGE','POTHOLES','GARBAGE','CONSTRUCTION_ROAD','BROKEN_SIGNAGE','BAD_STREETLIGHT','BAD_BILLBOARD','SAND_ON_ROAD','CLUTTER_SIDEWALK','UNKEPT_FACADE']
	#Get all the detected regions outputted by YOLO model
	texts=glob.glob(txtfilepath)
	with open(csvFileName,'w') as file:
		writer=csv.writer(file)
		writer.writerow(header)
		for c in range(len(texts)):
			files= pd.read_csv(texts[c],header=None)
			count=0
			#denormalise the coordinates		
			for i in range(len(files)):
				image_path = image_path1+texts[c].split('/')[5].split('.')[0]+".jpg"
				image = Image.open(image_path)
				image=np.asarray(image)
				x=int(float(files[0][i].split()[1])*1920)
				y=int(float(files[0][i].split()[2])*1080)
				w=int(float(files[0][i].split()[3])*1920)
				h=int(float(files[0][i].split()[4])*1080)
				y0=0
				y1=0
				x0=0
				x1=0
				if int(y-(h/2))<0:
					y0=0
				else:
					y0=int(y-(h/2))
			if int(y+int(h/2))>1079:
					y1=1079
				else:
					y1=int(y+int(h/2))
				if int(x-int(w/2))<0:
					x0=0
				else:
					x0=int(x-int(w/2))
				if int(x+int(w/2))>1919:
					x1=1919 
				else:
					x1=	int(x+int(w/2))
				if x0==x1 or y0==y1:
					count=count+1
				else:
					#save images for the detected regions
					annotation=image[y0:y1,x0:x1,:]
					im = Image.fromarray(annotation)
					im.save(imPath+str(i)+"_"+texts[c].split('/')[5].split('.')[0]+".jpg")
					clas=int(float(files[0][i].split()[0]))
					filename=texts[c].split('/')[5].split('.')[0]+".jpg"
					name=category[clas]
					id=str(i)
					t=[id,filename,name,str(x1),str(x0),str(y1),str(y0)]
					writer.writerow(t)
	file.close()