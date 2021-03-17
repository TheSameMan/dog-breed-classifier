# Dog Breed Classifier
Web application for classification of dog breeds by photo.

## About
The project consists of two parts: a study of neural network models for classifying dogs and a web application to demonstrate the work of the classifier. 

## Tech Stack
* PyTorch
* Flask
* Pillow

## Project Structure
```sh
├── model_development.ipynb *** Research
├── app.py
├── model.py *** Using mobilenet_model.pt
├── static
│	└── img
├── templates
│	└── predict.html
├── mobilenet_model.pt
├── resnet_model.pt
└── dogs *** Class names

```
---

---

## Deep Learning Models

### 1. Data
[Dataset](https://s3-us-west-1.amazonaws.com/udacity-aind/dog-project/dogImages.zip) contains about 8 thousand images of 133 classes of dogs and is divided into training, validation and test. 

### 2. Architecture
The size of dataset isn't enough to obtain a high accuracy of model. Therefore, transfer learning of the ResNet101 and MobileNet v3 models is used. In both models batch normalization is added before the last layer for faster convergence. In the MobileNet model, the dropout of the last layer is't used. Only normalization  and last layers are trained. Overfitting is compensated by data augmentation. 

### 3. Learning
The table below summarizes the learning parameters. 

| Parameter | ResNet | MobileNet |
|----------|:-------------:|------:|
| Top |  BatchNorm1d + Linear | BatchNorm1d + Linear |
| Weights Init | Standard | All Zeros |
| Algorithm | Adam | Adam |
| Learning Rate | 1e-3 | 1e-3 |
| Epochs | 12 | 100 |
| Batch Size | 256 | 256 |
| Loss Function | CrossEntropyLoss | CrossEntropyLoss |
| Scheduler | MultiStepLR | Custom |


### 4. Results
The table below shows the model test results and the size of the model file on disk. It can be seen that the accuracy of the ResNet model exceeds the accuracy of the MobileNet model due to the use "thicker" architecture. The demo application uses MobileNet as its size is much smaller and the accuracy is comparable.

| Parameter | ResNet | MobileNet |
|----------|:-------------:|------:|
| Accuracy | 91% | 82.3% |
| Disk Size | 333 MiB | 17 MiB |

---

---
## Web application Instructions
1. Install the dependencies:  
  ```
  $ pip install -r requirements.txt
  ``` 

2. Switch to project folder:  
  ```
  $ cd /path/to/dog_breed_classifier/
  ```

3. Run local server with debug:  
  ```
  $ FLASK_APP=app.py FLASK_DEBUG=1 flask run
  ```

4. Navigate to home page [http://127.0.0.1:5000/](http://127.0.0.1:5000/)
