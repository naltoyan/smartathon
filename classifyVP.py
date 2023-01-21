from keras.applications.inception_resnet_v2 import preprocess_input
from keras.models import Model, load_model
from keras_preprocessing.image import ImageDataGenerator

import numpy as np
import pandas as pd
import csv

#This function will classify the detected VP into one of the 10 classes
def classify_detectedVP(csvFileName,resultsFile,imPath,modelfile)
	#csvFileName is the desired name for the info csv file(output from get_detectedVP function), in our test is '/dataset/testsetPatches.csv'
	#resultsFile is the name of output csv file that contain coordinates of detected regions and their class
	#imPath is the path for the desired folder to save the extracted images, in our test is'/dataset/testset/'
	#modelfile is the .h5 trained model file, in our test is '/model/classify.h5'
	testdf=pd.read_csv(csvFileName,dtype=str)
	testdf['image_path']=testdf['id']+'_'+testdf['image_path']
	classes=10


	#Load inception_resnet_v2 trained model
	model = load_model(modelfile)



	#load the test patches
	test_datagen= ImageDataGenerator(preprocessing_function=preprocess_input) 	
	test_generator=test_datagen.flow_from_dataframe(dataframe=testdf, directory=imPath,x_col="image_path",y_col=None,batch_size=2,seed=42,shuffle=False,class_mode=None,target_size=(256,256)) 
	test_generator.reset()
	
	#Predict the class of detected VP
	preds=model.predict_generator(test_generator, steps=20117, verbose=0)
	predicted_class_indices=np.argmax(preds,axis=1)
	labels ={'0': 0, '1': 1, '10': 2, '2': 3, '3': 4, '4': 5, '5': 6, '7': 7, '8': 8, '9': 9}
	labels = dict((v,k) for k,v in labels.items())
	predictions = [labels[k] for k in predicted_class_indices]

	#Names of classes
	category=['GRAFFITI','FADED_SIGNAGE','POTHOLES','GARBAGE','CONSTRUCTION_ROAD','BROKEN_SIGNAGE','BAD_STREETLIGHT','BAD_BILLBOARD','SAND_ON_ROAD','CLUTTER_SIDEWALK','UNKEPT_FACADE']
	
	#Write the output CSV file
	header=['class','image_path','name','xmax','xmin','ymax','ymin']
	with open(resultsFile,'w') as file:
		writer=csv.writer(file)
		writer.writerow(header)
		filenames=test_generator.filenames
		for p in range(len(filenames)):
			t=[predictions[p], filenames[p].split('_')[1],category[int(predictions[p])],testdf['xmax'][p],testdf['xmin'][p],testdf['ymax'][p],testdf['ymin'][p]]
			writer.writerow(t)
	file.close()
	
	