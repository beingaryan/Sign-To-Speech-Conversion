
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
</br>
## :innocent: Motivation
A language interpreter is used by deaf people to seek help for __translating__ their thoughts to normal people. There’s a need for a system to __automatically recognize the sign language gestures__, which will lead to the minimization of the gap between deaf people and ordinary people in the society. Communication in sign language involves body movements, especially of hand, arms and some facial expressions.  people are not able to decode such gestures. </br></br>
Lack of efficient gesture detection system designed specifically for the differently abled, motivates us as a team to do something great in this field. The proposed work aims at converting such sign gestures into speech that can be understood by normal people. The system is trained by convolutional neural networks for the classification of __26 alphabets__ and one extra alphabet for null character. The proposed work has achieved an efficiency of __99.88%__ .




<!---Unable to communicate verbally is a disability. In order to communicate there are many ways, one of the most popular methods is the use of predefined sign languages. The purpose of this project is to bridge the __research gap__ and to contribute to recognize __American sign languages(ASL)__ with maximum efficiency. This repository focuses on the recognition of ASL in real time, converting predicted characters to sentences and output is generated in terms of voice formats. The system is trained by convolutional neural networks for the classification of __26 alphabets__ and one extra alphabet for null character. The proposed work has achieved an efficiency of __99.88%__ on the test set.--->


<p align="center"><img src="https://github.com/beingaryan/Sign-To-Speech-Conversion/blob/master/Analysis/WOW.jpg" width="700" height="400"></p>

## :warning: TechStack/framework used

- [OpenCV](https://opencv.org/)
- [Keras](https://keras.io/)
- [TensorFlow](https://www.tensorflow.org/)


## :file_folder: Data Distribution
The dataset used can be downloaded here - [Click to Download](https://drive.google.com/drive/folders/16ce6Hc4U5Qr6YBArcozoYom6TT5-7oSc?usp=sharing)

This dataset consists of __17113 images__ belonging to 27 classes:
*	__Training Set: 12845 images__<br />
<p align="center"><img src="https://github.com/beingaryan/Sign-To-Speech-Conversion/blob/master/Analysis/train_data_distribution.png" ></p>
<!---<br />![](Analysis/train_data_distribution.png)<br />--->
<p align="center">
  <b>Train Data Statistics</b></p>
*	__Test Set: 4368 images__<br />
<p align="center"><img src="https://github.com/beingaryan/Sign-To-Speech-Conversion/blob/master/Analysis/test_data_distribution.png" ></p>
<!---<br />![](Analysis/train_data_distribution.png)<br />--->
<p align="center">
  <b>Test Data Statistics</b></p>

## :star: Features
Our model is capable of predicting gestures from American sign language in real-time with high efficiency. These __'predicted alphabets'__ are converted to form __'words'__ and hence forms __'sentences'__. These sentences are converted into __'voice modules'__ by incorporating __'Google Text to Speech'__(gTTS API).</br></br>
The model is efficient, since we used a compact __'CNN-based architecture'__, it’s also computationally efficient and thus making it easier to deploy the model to embedded systems (Raspberry Pi, Google Coral, etc.). This system can therefore be used in real-time applications which aims at bridging the the gap in the process of communication between the __'Deaf and Dumb people with rest of the world'__.

## 🎨 Feature Extraction
* A Gaussian filter is applied to make the image smooth and  remove the noise.
* Intensity gradients of the image are calculated.
* Non-maximum suppression is applied to remove the possibility of a false response. 
* Double thresholding is done to detect or determine the possible edges. 
* Edges are finalized by identifying and removing all other edges that are weak and not linked to strong edges.<br />
<br />![](Analysis/fe.png)<br />
The above figure shows pre-processed image with extracted features which is sent to the model for classification.

## 🎯 Proposed Flow
</br>
<p align="center"><img src="https://github.com/beingaryan/Sign-To-Speech-Conversion/blob/master/Analysis/Proposed%20Flow.png" width="700" height="400"></p>
<!---![](Analysis/Proposed%20Flow.png)<br />--->
The figure above shows a detailed pipeline of the model architecture. It can be interpreted that a Convolutional architecture has been proposed.


## :key: Prerequisites

All the dependencies and required libraries are included in the file <code>requirements.txt</code> [See here](https://github.com/beingaryan/Sign-To-Speech-Conversion/blob/master/requirements.txt)

## 🚀&nbsp; Installation
1. Clone the repo
```
$ git clone https://github.com/beingaryan/Sign-To-Speech-Conversion.git
```

2. Change your directory to the cloned repo and create a Python virtual environment named 'test'
```
$ mkvirtualenv test
```

3. Now, run the following command in your Terminal/Command Prompt to install the libraries required
```
$ pip3 install -r requirements.txt
```

## :bulb: Working

1. Open terminal. Go into the cloned project directory and type the following command:
```
$ python3 jupyter
```

2. To train the model, open the [ASL_train](https://github.com/beingaryan/Sign-To-Speech-Conversion/blob/master/ASL_train.ipynb) file in jupyter notebook and run all the cells </br>

3. To detect ASL Gestures in real-time video streams run the [ASL_Real-Time.ipynb](https://github.com/beingaryan/Automated-Sign-To-Speech-Conversion/blob/master/ASL_Real-Time.ipynb) file.

* __'NOTE'__: You can directly use [asl_classifier.h5](https://github.com/beingaryan/Automated-Sign-To-Speech-Conversion/blob/master/asl_classifier.h5) file trained by me for real-time predictions in [Real-Time](https://github.com/beingaryan/Automated-Sign-To-Speech-Conversion/blob/master/ASL_Real-Time.ipynb) file.

</br></br>
## :key: Results 
#### Our model gave 99.8% accuracy for Sign Language Detection after training via <code>tensorflow-gpu==2.0.0</code>
<br /><br />![](Analysis/Classification_report.png)<br />
* The model has been trained on a python based environment on Jupyter platform for 20 epochs. 
* The model has achieved an accuracy of __97.45 %__ on Training Set with __99.88 %__ accuracy on the Validation set.
* The prescribed model has been evaluated on __Test set__ where it has attained an accuracy of __99.85%__ with loss of 0.60 %.
#### We got the following accuracy/loss training curve plot
![](Analysis/Loss%20vs%20Epoch.png)<br />
#### The above figure shows the Loss plot of the model throughout it's training journey. 

<br /><br />![](Analysis/Accuracy%20vs%20Epoch.png)<br/>
#### The above figure shows the Accuracy plot of the model throughout it's training journey. 

## 📈 Analysis

The training phase data has been evaluated on the Heatmap Plotting representation. 
![](Analysis/Heatmp.png)<br />
#### The prescribed inference suggests that the trained classes are highly correlated with the same class of the data. 

## :clap: And it's done!
Feel free to mail me for any doubts/query 
:email: aryan.gupta18@vit.edu



## :handshake: Contribution
Feel free to **file a new issue** with a respective title and description on the the [Sign-Language-Detection](https://github.com/beingaryan/Sign-To-Speech-Conversion/issues) repository. If you already found a solution to your problem, **I would love to review your pull request**!


## :heart: Owner
Made with :heart:&nbsp;  by [Aryan Gupta](https://github.com/beingaryan)


## :+1: Credits
* [https://www.pyimagesearch.com/](https://www.pyimagesearch.com/)
* [https://opencv.org/](https://opencv.org/)
* K. Manikandan, Ayush Patidar, Pallav Walia, Aneek Barman Roy.(2017). Hand Gesture Detection and Conversion to Speech and Text. ARXIV
* Omkar Vedak, Prasad Zavre, Abhijeet Todkar, Manoj Patil.(2019). INTERNATIONAL RESEARCH JOURNAL OF ENGINEERING AND TECHNOLOGY (IRJET).Sign Language Interpreter using Image Processing and Machine Learning
* Pramada, Sawant & Vaidya, Archana. (2013). Intelligent Sign Language Recognition Using Image Processing. IOSR Journal of Engineering. 03. 45-51. 10.9790/3021-03224551.

## :handshake: Our Contributors

[CONTRIBUTORS.md](/CONTRIBUTORS.md)

## :eyes: Code of Conduct

You can find our Code of Conduct [here](/CODE_OF_CONDUCT.md).


## :eyes: License
MIT © [Aryan Gupta](https://github.com/beingaryan/Sign-To-Speech-Conversion/blob/master/LICENSE)








