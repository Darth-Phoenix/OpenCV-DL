from tkinter import image_names
from PyQt5 import QtWidgets, QtGui, QtCore
from UI import Ui_MainWindow
import os
import cv2
import numpy as np

class MainWindow_controller(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__() 
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)
        self.setup_control()

    def setup_control(self):
        self.ui.LoadImageButton.clicked.connect(self.loadImage)
        self.ui.pushButton_1_1.clicked.connect(self.gaussianBlur)
        self.ui.pushButton_1_2.clicked.connect(self.sobelX)
        self.ui.pushButton_1_3.clicked.connect(self.sobelY)
        self.ui.pushButton_1_4.clicked.connect(self.magnitude)
        self.ui.pushButton_2_1.clicked.connect(self.resizeImg)
        self.ui.pushButton_2_2.clicked.connect(self.translation)
        self.ui.pushButton_2_3.clicked.connect(self.rotationScaling)
        self.ui.pushButton_2_4.clicked.connect(self.shearing)

    def loadImage(self):
        filePath, _ = QtWidgets.QFileDialog.getOpenFileName(None, 'OpenFile', '', "Image file(*.jpg *.png)")
        _, fileName = os.path.split(filePath)
        self.ui.ImageText.setText(fileName)
        self.img = cv2.imread(filePath)    
    
    def convolution(self, image, kernel):
        output = np.zeros_like(image)
        image_padded = np.zeros((image.shape[0]+2, image.shape[1]+2))
        image_padded[1:-1, 1:-1] = image

        for i in range(image.shape[0]):
            for j in range(image.shape[1]):
                output[i, j] = np.sum(kernel * image_padded[i: i+3, j: j+3])

        return output

    def gaussianBlur_func(self):
        gray = cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY)
        filter = np.fromfunction(lambda i, j: np.exp(-((i-1)*(i-1) + (j-1)*(j-1))) / np.pi, (3, 3), dtype = np.float32)
        filter = filter / filter.sum()
        image = self.convolution(gray, filter)
        return image

    def sobelX_func(self):
        gaussian_image = np.float32(self.gaussianBlur_func())
        sobelX_filter = np.float32([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
        sobelX_image = np.uint8(np.abs(self.convolution(gaussian_image, sobelX_filter)))
        return sobelX_image

    def sobelY_func(self):
        gaussian_image = np.float32(self.gaussianBlur_func())
        sobelY_filter = np.float32([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])
        sobelY_image = np.uint8(np.abs(self.convolution(gaussian_image, sobelY_filter)))
        return sobelY_image

    def gaussianBlur(self):
        result = self.gaussianBlur_func()
        cv2.imshow("Gaussian Blur", result)

    def sobelX(self):
        result = self.sobelX_func()
        cv2.imshow("Sobel X", result)

    def sobelY(self):
        result = self.sobelY_func()
        cv2.imshow("Sobel Y", result)

    def magnitude(self):
        sobelX_image = np.float32(self.sobelX_func())
        sobelY_image = np.float32(self.sobelY_func())
        magnitude = (sobelX_image**2 + sobelY_image**2)**0.5

        magnitude *= 255.0 / magnitude.max()
        result = np.uint8(magnitude)
        cv2.imshow("Magnitude", result)

    def resizeImg(self):
        image = cv2.resize(self.img, (215, 215), cv2.INTER_AREA)
        M = np.float32([[1, 0, 0], [0, 1, 0]]) 
        result = cv2.warpAffine(image, M, (430, 430))
        self.img = result
        cv2.imshow("Resize", result)

    def translation(self):
        img1 = self.img
        M = np.float32([[1, 0, 215], [0, 1, 215]]) 
        img2 = cv2.warpAffine(self.img, M, (430, 430))
        result = cv2.addWeighted(img1, 1, img2, 1, 0)
        self.img = result
        cv2.imshow("Translation", result)

    def rotationScaling(self):
        M = cv2.getRotationMatrix2D((215, 215), 45, 0.5)
        result = cv2.warpAffine(self.img, M, (430, 430))
        self.img = result
        cv2.imshow("Rotation, Scaling", result)

    def shearing(self):
        old = np.float32([[50, 50], [200, 50], [50, 200]])
        new = np.float32([[10, 100], [100, 50], [100, 250]])
        M = cv2.getAffineTransform(old, new)
        result = cv2.warpAffine(self.img, M, (430, 430))
        self.img = result
        cv2.imshow("Shearing", result)
        





                



