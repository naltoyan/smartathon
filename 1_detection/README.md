# SMARTATHON
 
This is the repository for 5 group members participating in a hackathon: SMARTATHON.


The "Submission" folder contains the material for the final submission.

To Run the model on the test dataset:

1- prepare the test images and get them from the folder 'images':

< pip install -qr requirements.txt >
 < python get_test.py get_images ./dataset test_dataset >

2- train the model [no need it for test]:

The code has the source code that used to build the yolov5 model and train it on the dataset. 
< cd yolov5 && python train.py --img 1280 --batch 12 --epochs 90 --data './smartathon.yaml' --weights 'yolov5x.pt' --name 'result' --optimizer Adam --workers 12 --hyp './hyp.yaml' >

The result model of our training, we call: 'SMARTATHON.pt'

3- run our model 'SMARTATHON.pt' on the test dataset:

 < python yolov5/detect.py --weights './SMARTATHON.pt' --data './smartathon.yaml' --conf 0.1 --imgsz 1280 --source './test_dataset/test/images/' --save-txt --save-conf --save-crop --project ./detection_results >


4- the output will be in : 'detection_results/exp'

4- to run the classifier: .....