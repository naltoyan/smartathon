import get_detectedVP
import classify

#test the proposed solution
txtfilepath='/labels/*.txt'
csvFileName='/dataset/testsetPatches.csv'
imPath='/dataset/testset/'
image_path1='/dataset/images/'
resultsFile='/dataset/results.csv'
modelfile='/model/classify.h5'



#Classify the detected VP under one of the 10 classes (DEEMA)
detectedVP.get_detectedVP(txtfilepath,csvFileName,imPath,image_path1)
classify.classify_detectedVP(csvFileName,resultsFile,imPath,modelfile)
	


