from tkinter.tix import IMAGE
from PyQt5 import QtWidgets, QtGui, QtCore
from UI import Ui_MainWindow
import os
import cv2
import numpy as np
import torch
import torchvision
from torchsummary import summary
import torchvision.transforms as transforms
import random
import matplotlib.pyplot as plt
import urllib.request
import ssl



class MainWindow_controller(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__() 
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)
        self.setup_control()

        ssl._create_default_https_context = ssl._create_unverified_context
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))]) 
        self.trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform = self.transform)
        self.trainloader = torch.utils.data.DataLoader(self.trainset, batch_size=16, shuffle=True, num_workers=2)
        self.testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform = self.transform)
        self.testloader = torch.utils.data.DataLoader(self.testset, batch_size=16, shuffle=False, num_workers=2)
        self.net = torchvision.models.vgg19().to(self.device)
        self.classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    def setup_control(self):
        self.ui.LoadImageButton.clicked.connect(self.loadImage)
        self.ui.pushButton_1_1.clicked.connect(self.showImage)
        self.ui.pushButton_1_2.clicked.connect(self.modelStructure)
        self.ui.pushButton_1_3.clicked.connect(self.augmentation)
        self.ui.pushButton_1_4.clicked.connect(self.showAccuracy)
        self.ui.pushButton_1_5.clicked.connect(self.inference)

    def loadImage(self):
        filePath, _ = QtWidgets.QFileDialog.getOpenFileName(None, 'OpenFile', '', "Image file(*.jpg *.png)")
        _, fileName = os.path.split(filePath)
        self.ui.ImageText.setText(fileName)
        self.img = cv2.imread(filePath)
    
    def showImage(self):
        rand_idx = random.sample(range(50000), 9)
        plt.figure(num = 'Figure 1')
        idx = 1
        for i in rand_idx:
            plt.subplot(3, 3, idx)
            plt.title(self.classes[self.trainset.targets[i]])
            plt.imshow(self.trainset.data[i, :])
            plt.axis('off')
            idx+=1
        plt.show()

    def modelStructure(self):
        summary(self.net, (3, 224, 224)) 

    def augmentation(self):
        plt.figure(num='Figure 1')
        trans_toPIL = transforms.ToPILImage()
        image = trans_toPIL(self.img)

        plt.subplot(1, 3, 1)
        augmentation = transforms.Compose([transforms.RandomRotation(90)])
        result = augmentation(image)
        plt.title('RandomRotation')
        plt.imshow(result)
        plt.axis('off')
        
        plt.subplot(1, 3, 2)
        augmentation = transforms.Compose([transforms.RandomResizedCrop(size = self.img.shape[:2], scale=(0.08, 1.0), ratio=(0.75, 1.33))])
        result = augmentation(image)
        plt.title('RandomResizedCrop')
        plt.imshow(result)
        plt.axis('off')
        
        plt.subplot(1, 3, 3)
        augmentation = transforms.Compose([transforms.RandomHorizontalFlip(0.5)])
        result = augmentation(image)
        plt.title('RandomHorizontalFlip')
        plt.imshow(result)
        plt.axis('off')

        plt.show()

    def showAccuracy(self):
        filePath = './accuracy_loss.png'
        result = cv2.imread(filePath)
        cv2.imshow("Model analysis", result)

    def inference(self):
        model = self.net
        model.load_state_dict(torch.load('cifar_net.pth'))
        model.eval()
        trans_toPIL = transforms.ToPILImage()
        image = trans_toPIL(self.img)
        augmentation = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor(), transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))]) 
        image = augmentation(image)
        image = image.to(self.device)
        with torch.no_grad():
            predict = model(image)
            print(predict)

    
        





                



