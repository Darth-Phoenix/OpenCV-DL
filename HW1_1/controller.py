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
        self.ui.LoadImageButton1.clicked.connect(self.loadImage1)
        self.ui.LoadImageButton2.clicked.connect(self.loadImage2)
        self.ui.pushButton_1_1.clicked.connect(self.colorSeperate)
        self.ui.pushButton_1_2.clicked.connect(self.colorTransform)
        self.ui.pushButton_1_3.clicked.connect(self.colorDetection)
        self.ui.pushButton_1_4.clicked.connect(self.blending)
        self.ui.pushButton_2_1.clicked.connect(self.gaussianBlur)
        self.ui.pushButton_2_2.clicked.connect(self.bilateralFilter)
        self.ui.pushButton_2_3.clicked.connect(self.medianFilter)

    def loadImage1(self):
        filePath, _ = QtWidgets.QFileDialog.getOpenFileName(None, 'OpenFile', '', "Image file(*.jpg)")
        _, fileName = os.path.split(filePath)
        self.ui.ImageText1.setText(fileName)
        self.img1 = cv2.imread(filePath)

    def loadImage2(self):
        filePath, _ = QtWidgets.QFileDialog.getOpenFileName(None, 'OpenFile', '', "Image file(*.jpg)")
        _, fileName = os.path.split(filePath)
        self.ui.ImageText2.setText(fileName)
        self.img2 = cv2.imread(filePath)

    def colorSeperate(self):
        (B, G, R) = cv2.split(self.img1)
        zeros = np.zeros(self.img1.shape[:2], dtype='uint8')
        imgB = cv2.merge([B, zeros, zeros])
        imgG = cv2.merge([zeros, G, zeros])
        imgR = cv2.merge([zeros, zeros, R])
        cv2.imshow("B channel", imgB)
        cv2.imshow("G channel", imgG)
        cv2.imshow("R channel", imgR)

    def colorTransform(self):
        gray = cv2.cvtColor(self.img1, cv2.COLOR_BGR2GRAY)
        cv2.imshow("I1", gray)
        (B, G, R) = cv2.split(self.img1)
        for i in range(self.img1.shape[0]):
            for j in range(self.img1.shape[1]):
                gray[i][j] = (int(B[i][j]) + int(G[i][j]) + int(R[i][j])) / 3
        cv2.imshow("I2", gray)
    
    def colorDetection(self):
        hsv = cv2.cvtColor(self.img1, cv2.COLOR_BGR2HSV)
        lowerG = np.array([40, 50, 20], np.uint8)
        upperG = np.array([80, 255, 255], np.uint8)
        lowerW = np.array([0, 0, 200], np.uint8)
        upperW = np.array([180, 20, 255], np.uint8)
        mask = cv2.inRange(hsv , lowerG , upperG)
        result = cv2.bitwise_and(self.img1, self.img1, mask=mask)
        cv2.imshow("I1", result)
        mask = cv2.inRange(hsv, lowerW, upperW)
        result = cv2.bitwise_and(self.img1, self.img1, mask=mask)
        cv2.imshow("I2", result)

    def onBlend(self, val):
        beta = val / 255
        alpha = 1 - beta
        result = cv2.addWeighted(self.img1, alpha, self.img2, beta, 0)
        cv2.imshow("Blend", result)

    def blending(self):
        cv2.namedWindow("Blend")
        cv2.resizeWindow("Blend", self.img1.shape[1], self.img1.shape[0])
        cv2.createTrackbar("Blend", "Blend", 0, 255, self.onBlend)
        cv2.imshow("Blend", self.img1)
    
    def onBlur(self, m):
        image = cv2.GaussianBlur(self.img1, (2*m+1, 2*m+1), 0)
        cv2.imshow("gaussian_blur", image)

    def gaussianBlur(self):
        cv2.namedWindow("gaussian_blur")
        cv2.resizeWindow("gaussian_blur", self.img1.shape[1], self.img1.shape[0])
        cv2.createTrackbar("magnitude", "gaussian_blur", 0, 10, self.onBlur)
        cv2.imshow("gaussian_blur", self.img1)

    def onBilateral(self, m):
        image = cv2.bilateralFilter(self.img1, 2*m+1, 90, 90)
        cv2.imshow("bilateral_filter", image)
    
    def bilateralFilter(self):
        cv2.namedWindow("bilateral_filter")
        cv2.resizeWindow("bilateral_filter", self.img1.shape[1], self.img1.shape[0])
        cv2.createTrackbar("magnitude", "bilateral_filter", 0, 10, self.onBilateral)
        cv2.imshow("bilateral_filter", self.img1)

    def onMedian(self, m):
        image = cv2.medianBlur(self.img1, 2*m+1)
        cv2.imshow("median_filter", image)
    
    def medianFilter(self):
        cv2.namedWindow("median_filter")
        cv2.resizeWindow("median_filter", self.img1.shape[1], self.img1.shape[0])
        cv2.createTrackbar("magnitude", "median_filter", 0, 10, self.onMedian)
        cv2.imshow("median_filter", self.img1)

