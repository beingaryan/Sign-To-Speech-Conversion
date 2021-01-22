
<h1 align="center">Sign Language to Speech Conversion</h1>

<div align= "center">
  <h4>Sign Language to Speech Conversion system built with OpenCV, Keras/TensorFlow using Deep Learning and Computer Vision concepts in order to communicate using American Sign Language(ASL) based gestures in real-time video streams with differently abled. </h4>
</div>

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
![Python](https://img.shields.io/badge/python-v3.6+-blue.svg)
[![contributions welcome](https://img.shields.io/badge/contributions-welcome-brightgreen.svg?style=flat)](https://github.com/beingaryan/Sign-To-Speech-Conversion/issues)
[![Forks](https://img.shields.io/github/forks/beingaryan/Sign-To-Speech-Conversion.svg?logo=github)](https://github.com/beingaryan/Sign-To-Speech-Conversion/network/members)
[![Stargazers](https://img.shields.io/github/stars/beingaryan/Sign-To-Speech-Conversion.svg?logo=github)](https://github.com/beingaryan/Sign-To-Speech-Conversion/stargazers)
[![Issues](https://img.shields.io/github/issues/beingaryan/Sign-To-Speech-Conversion.svg?logo=github)](https://github.com/beingaryan/Sign-To-Speech-Conversion/issues)
[![LinkedIn](https://img.shields.io/badge/-LinkedIn-black.svg?style=flat-square&logo=linkedin&colorB=555)](https://www.linkedin.com/in/aryan-gupta-6a9201191/)

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
![Live Demo](Analysis/output.gif)

## :point_down: Support me here!
<a href="https://www.buymeacoffee.com/beingaryan" target="_blank"><img src="https://www.buymeacoffee.com/assets/img/custom_images/orange_img.png" alt="Buy Me A Coffee" style="height: 41px !important;width: 174px !important;box-shadow: 0px 3px 2px 0px rgba(190, 190, 190, 0.5) !important;-webkit-box-shadow: 0px 3px 2px 0px rgba(190, 190, 190, 0.5) !important;" ></a>

## :innocent: Motivation
Unable to communicate verbally is a disability. In order to communicate there are many ways, one of the most popular methods is the use of predefined sign languages. The purpose of this project is to bridge the __'research gap'__ and to contribute to recognize __'American sign languages(ASL)'__ with maximum efficiency. This repository focuses on the recognition of ASL in real time, converting predicted characters to sentences and output is generated in terms of voice formats. The system is trained by convolutional neural networks for the classification of __'26 alphabets'__ and one extra alphabet for null character. The proposed work has achieved an efficiency of __'99.88%'__ on the test set.


<p align="center"><img src="https://github.com/beingaryan/Sign-To-Speech-Conversion/blob/master/Analysis/asl.jpg" width="700" height="400"></p>

## :warning: TechStack/framework used

- [OpenCV](https://opencv.org/)
- [Keras](https://keras.io/)
- [TensorFlow](https://www.tensorflow.org/)


## :file_folder: Data Distribution
The dataset used can be downloaded here - [Click to Download](https://drive.google.com/drive/folders/16ce6Hc4U5Qr6YBArcozoYom6TT5-7oSc?usp=sharing)

This dataset consists of __17113 images__ belonging to 27 classes:
*	__Training Set: 12845 images__<br />
<br />![](Analysis/train_data_distribution.png)<br />
The figure above shows the training data distribution.<br />
*	__Test Set: 4368 images__<br />
<br />![](Analysis/train_data_distribution.png)<br />
The figure above shows the test data distribution.<br />
<br />

## Feature Extraction
* A Gaussian filter is applied to make the image smooth and  remove the noise.
* Intensity gradients of the image are calculated.
* Non-maximum suppression is applied to remove the possibility of a false response. 
* Double thresholding is done to detect or determine the possible edges. 
* Edges are finalized by identifying and removing all other edges that are weak and not linked to strong edges.<br />
<br />![](Analysis/fe.png)<br />
The above figure shows pre-processed image with extracted features which is sent to the model for classification.
## Proposed Flow

![](Analysis/Proposed%20Flow.png)<br />
The figure above shows a detailed pipeline of the model architecture. It can be interpreted that a Convolutional architecture has been proposed.

## SETUP
* Fork the repository at your profile
* Git Clone the repository to your local machine. 
* Fire the command: ```pip install - r requirements.txt```
* Download the dataset from the mentioned [LINK](https://drive.google.com/drive/folders/16ce6Hc4U5Qr6YBArcozoYom6TT5-7oSc?usp=sharing).
* Load the Dataset and the Training file from [Train_File](https://github.com/beingaryan/Automated-Sign-To-Speech-Conversion/blob/master/ASL_train.ipynb).
* Predict Real-Time Sentences using [Real-Time](https://github.com/beingaryan/Automated-Sign-To-Speech-Conversion/blob/master/ASL_Real-Time.ipynb) file.
* NOTE: You can directly use [asl_classifier.h5](https://github.com/beingaryan/Automated-Sign-To-Speech-Conversion/blob/master/asl_classifier.h5) file trained by me for real-time predictions in [Real-Time](https://github.com/beingaryan/Automated-Sign-To-Speech-Conversion/blob/master/ASL_Real-Time.ipynb) file.

## Results and Analysis
* The model has been trained on a python based environment on Jupyter platform for 20 epochs. 
* The model has achieved an accuracy of 97.45 % on Training Set with 99.88 % accuracy on the Validation set.
* The prescribed model has been evaluated on Test set where it has attained an accuracy of 99.85% with loss of 0.60 %.
![](Analysis/Loss%20vs%20Epoch.png)<br />
* The above figure shows the Loss plot of the model throughout it's training journey. 
* It can be interpreted that the loss decreases with increasing epochs.
<br /><br />![](Analysis/Accuracy%20vs%20Epoch.png)<br/>
* The above figure shows the Accuracy plot of the model throughout it's training journey. 
* It can be interpreted that the accuracy incraeses with increasing epochs for both train and val sets.
<br /><br /><br />
* The training phase data has been evaluated on the Heatmap Plotting representation. 
![](Analysis/Heatmp.png)<br />
* The prescribed inference suggests that the trained classes are highly correlated with the same class of the data. 


* The model has been evaluated for Precision, Recall, F1-score metrics for all the 26 classes along with a null class. 
<br /><br />![](Analysis/Classification_report.png)<br />
* The analysis carried has been shown in the classification report attached above. 
* It can be interpreted that the average weighted F1-score metrics is 1 which describes effective learning and low false predictions.

## Output Snapshots
![](Analysis/WOW.jpg)<br />
The above figure shows correctly classified word: "WOW"<br /> 
![](Analysis/I%20SEE%20IT.png)<br />
The above figure shows correctly classified word: "I SEE IT" 
![](Analysis/HI%20HOW%20ARE%20YOU.jpg)<br />
The above figure shows correctly classified word: "HI HOW ARE YOU" 
<br />
## References
The above work is inspired by several recent researches in the field of Deep Learning. Some notable shoutout goes to:<br />
* K. Manikandan, Ayush Patidar, Pallav Walia, Aneek Barman Roy.(2017). Hand Gesture Detection and Conversion to Speech and Text. ARXIV
* Omkar Vedak, Prasad Zavre, Abhijeet Todkar, Manoj Patil.(2019). INTERNATIONAL RESEARCH JOURNAL OF ENGINEERING AND TECHNOLOGY (IRJET).Sign Language Interpreter using Image Processing and Machine Learning
* Pramada, Sawant & Vaidya, Archana. (2013). Intelligent Sign Language Recognition Using Image Processing. IOSR Journal of Engineering. 03. 45-51. 10.9790/3021-03224551.

## Doubt Support
For further queries, you can reach out to me via:
* [Linkedin](https://www.linkedin.com/in/aryan-gupta-6a9201191/) handle
* [Instagram](https://www.instagram.com/beingryaan/)
* Email-Id: aryan.gupta18@vit.edu

## Contribution Norms
* Feel free to come up with updates and modifications in the project.
* Pull requests are welcomed.





