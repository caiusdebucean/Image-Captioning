# Image-Captioning
A Neural Network trained on MS-COCO to caption images. This is the second project of the [*Udacity Computer Vision Expert Nanodegree*](https://www.udacity.com/course/computer-vision-nanodegree--nd891).

### Architecture

The network uses a **CNN encoder** for image features and a **RNN decoder** for caption prediction. It uses ResNet-50 connected to a layer of LSTM with a hidden size and embedding size of _512_ .

![Overview](https://i.imgur.com/Oj8FHRw.png)

_The images are fet at the beggining, in the CNN Encoder, while the RNN Decoder takes in both the embedded image feature vector and the word embeddings._

### Results

These are the results after 3 epochs of training 

**Good results**

![Overview](https://i.imgur.com/qsYmtQ2.png)

**Bad results**

![Overview](https://i.imgur.com/JTPk5WI.png)

___

In order to duplicate these results, consider that the project is structured as a series of Jupyter **notebooks** that are designed to be completed in **sequential order** (`0_Dataset.ipynb, 1_Preliminaries.ipynb, 2_Training.ipynb, 3_Inference.ipynb`). Some code is give, but the core for understanding it is left blank. These notebooks are completed by my observations.


### Requirements
**1. Setup COCO API**

***MacOS/Linux***
1. Clone this repo: https://github.com/cocodataset/cocoapi  
```
git clone https://github.com/cocodataset/cocoapi.git  
```

2. Setup the COCO API (also described in the readme [here](https://github.com/cocodataset/cocoapi)) 
```
cd cocoapi/PythonAPI  
make  
cd ..
```
***Windows***
To install COCO API follow steps listed here: https://github.com/philferriere/cocoapi, a fork maintained by [philferriere](https://github.com/philferriere/cocoapi).

**2. Download Dataset**
Download some specific data from here: http://cocodataset.org/#download (described below)

* Under **Annotations**, download:
  * **2014 Train/Val annotations [241MB]** (extract captions_train2014.json and captions_val2014.json, and place at locations cocoapi/annotations/captions_train2014.json and cocoapi/annotations/captions_val2014.json, respectively)  
  * **2014 Testing Image info [1MB]** (extract image_info_test2014.json and place at location cocoapi/annotations/image_info_test2014.json)

* Under **Images**, download:
  * **2014 Train images [83K/13GB]** (extract the train2014 folder and place at location cocoapi/images/train2014/)
  * **2014 Val images [41K/6GB]** (extract the val2014 folder and place at location cocoapi/images/val2014/)
  * **2014 Test images [41K/6GB]** (extract the test2014 folder and place at location cocoapi/images/test2014/)

```sh
📂 [Project-Folder]
└ 📂 opt
  └ 📂 cocoapi
    └ 📂 annotations
      └ 📄 captions_train2014.json
      └ 📄 captions_val2014.json
      └ 📄 image_info_test2014.json
      └ 📄 instances_train2014.json
      └ 📄 instances_val2014.json
      └ 📄 person_keypoints_train2014.json
      └ 📄 person_keypoints_val2014.json
    └ 📂 images
      └ 📂 test2014
        └ 📄 COCO_test2014_000000000001.jpg 
        └ 📄 ...
      └ 📂 train2014
        └ 📄 COCO_train2014_000000000009.jpg
        └ 📄 ...
```

**3. Create your enviroment**

I recommend using [Conda](https://docs.conda.io/en/latest/), and installing all the packages with `conda install` or `pip install`

**4. Locate the pretrained Models (Optional)**
Pre-trained model trained for 3 epochs can be found in `models` folder.
```sh
📂 [Project-Folder]
└ 📂 weights
    └ 📄 encoder-3.pkl
    └ 📄 decoder-3.pkl
```


**IMPORTANT Observation**: If you want to work on the **notebooks**, copy them into the main project folder, where _model.py_ and the rest of the files are.

### Training

Considering all the requirements are met , run this command:

> python train.py

Modify the train parameters in the file if you want to test different ways.
_Note that this will automatically save the models at each epoch_

### Testing

Run the following command:

> python test.py

It makes use of two functions to cleanly print the predicted captions.
_For getting validation results with metrics, I recommend exporting the outputs of all validation set in a .json file and comparing it with the groundtruth results for each image index_


Debucean Caius-Ioan @Udacity 