from tkinter.tix import IMAGE
from PyQt5 import QtWidgets, QtGui, QtCore
from UI import Ui_MainWindow
import os
import cv2
import numpy as np
import random
import pathlib
import matplotlib.pyplot as plt
import tensorflow as tf
from keras.models import load_model
from keras import layers
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D
from keras.layers import MaxPooling2D, GlobalMaxPooling2D, Activation
from keras.preprocessing.image import ImageDataGenerator
from keras import optimizers
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
from keras.models import Model
from tensorflow.keras.applications.resnet50 import ResNet50
import tensorflow_addons as tfa



class MainWindow_controller(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__() 
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)
        self.setup_control()

    def setup_control(self):
        self.ui.LoadImageButton.clicked.connect(self.loadImage)
        self.ui.pushButton_1_1.clicked.connect(self.showImage)
        self.ui.pushButton_1_2.clicked.connect(self.distribution)
        self.ui.pushButton_1_3.clicked.connect(self.modelStructure)
        self.ui.pushButton_1_4.clicked.connect(self.comparison)
        self.ui.pushButton_1_5.clicked.connect(self.inference)

    def loadImage(self):
        filePath, _ = QtWidgets.QFileDialog.getOpenFileName(None, 'OpenFile', '', "Image file(*.jpg *.png)")
        _, fileName = os.path.split(filePath)
        self.ui.ImageText.setText(fileName)
        self.img = cv2.imread(filePath)
    
    def showImage(self):
        dir = './inference_dataset'
        classes = ['Cat', 'Dog']
        img_array = [[] for i in range(2)]
        for i in range(len(classes)):
            for filename in os.listdir(r"./"+dir+'/'+classes[i]):
                img = cv2.imread(dir + '/' + classes[i] + '/' + filename)
                img = cv2.resize(img, (224, 224))
                img_array[i].append(img)

        plt.figure(num = 'Figure 1')
        for i in range(len(classes)):
            plt.subplot(1,2, i+1)
            plt.title(classes[i])
            plt.imshow(cv2.cvtColor(random.choice(img_array[i]), cv2.COLOR_BGR2RGB))
            plt.axis('off')
        plt.show()
    
    def distribution(self):
        # dir = './training_dataset'
        # classes = ['Cat', 'Dog']
        # height = []
        # path_dir = pathlib.Path(dir+'/Cat')    
        # image_count = len(list(path_dir.glob('*.jpg')))
        # height.append(image_count)
        # path_dir = pathlib.Path(dir+'/Dog')    
        # image_count = len(list(path_dir.glob('*.jpg')))
        # height.append(image_count)
        # plt.figure(num = 'Figure 1')
        # plt.title('Class Distribution')
        # for a,b in enumerate(height):  
        #     plt.text(a, b+0.05, '%.0f' % b, ha='center', va= 'bottom',fontsize=11)     
        # plt.bar(classes, height)
        # plt.ylabel('Number of Images')
        # plt.show()
        filePath = './distribution.png'
        result = cv2.imread(filePath)
        cv2.imshow("Distribution", result)

    def modelStructure(self):
        net = ResNet50(include_top=False, weights='imagenet', input_shape=(224, 224, 3))
        x = net.output
        x = Flatten()(x)
        x = Dropout(0.5)(x)
        output_layer = Dense(1, activation='sigmoid')(x)
        model = Model(inputs=net.input, outputs=output_layer)
        model.summary()
    
    def comparison(self):
        # dir = './validation_dataset'
        # val_ds = tf.keras.utils.image_dataset_from_directory(dir, image_size = (224, 224), batch_size = 8)
        # classes = ['Binary Cross Entropy', 'Focal Loss']
        # height = []
        # model = load_model('./binary.model')
        # score = model.evaluate(val_ds)
        # height.append(score[1]*100)
        # model = load_model('./focal.model')
        # score = model.evaluate(val_ds)
        # height.append(score[1]*100)
        # plt.figure(num = 'Figure 1')
        # plt.title('Accuracy Comparison')
        # for a,b in enumerate(height):  
        #     plt.text(a, b+0.05, '%.2f' % b, ha='center', va= 'bottom',fontsize=11)     
        # plt.bar(classes, height)
        # plt.ylabel('Accuracy (%)')
        # plt.show()
        filePath = './comparison.png'
        result = cv2.imread(filePath)
        cv2.imshow("Comparison", result)

    def inference(self):
        classes = ['Cat', 'Dog']
        image = cv2.resize(self.img, (224, 224))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_tensor = image.reshape(1, 224, 224, 3)
        model = load_model('./binary.model')
        prediction = model.predict_step(image_tensor)
        prediction = prediction.numpy()
        prediction = prediction.astype(int)
        plt.figure(num = 'Inference')
        plt.title('Prediction: ' + classes[prediction[0][0]])
        plt.imshow(image)
        plt.axis('off')
        plt.show()
