from tkinter.tix import IMAGE
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
        self.ui.LoadFolderButton.clicked.connect(self.loadFolder)
        self.ui.LoadImageButton_L.clicked.connect(self.loadImage1)
        self.ui.LoadImageButton_R.clicked.connect(self.loadImage2)
        self.ui.pushButton_1_1.clicked.connect(self.drawContour)
        self.ui.pushButton_1_2.clicked.connect(self.countRings)
        self.ui.pushButton_2_1.clicked.connect(self.findCorners)
        self.ui.pushButton_2_2.clicked.connect(self.findIntrinsic)
        self.ui.pushButton_2_3.clicked.connect(self.findExtrinsic)
        self.ui.pushButton_2_4.clicked.connect(self.findDistortion)
        self.ui.pushButton_2_5.clicked.connect(self.undistort)
        self.ui.pushButton_3_1.clicked.connect(self.showWordsOnBoard)
        self.ui.pushButton_3_2.clicked.connect(self.showWordsVertically)
        self.ui.pushButton_4_1.clicked.connect(self.stereoDisparityMap)
        
    def loadFolder(self):
        folderpath = QtWidgets.QFileDialog.getExistingDirectory(None, 'Select Folder')
        self.ui.label_folder.setText(folderpath)
        self.folderpath = folderpath

    def loadImage1(self):
        filePath, _ = QtWidgets.QFileDialog.getOpenFileName(None, 'Open File', '', "Image file(*.jpg *.png)")
        _, fileName = os.path.split(filePath)
        self.ui.label_image1.setText(fileName)
        self.filename1 = fileName
        self.img1 = cv2.imread(filePath)

    def loadImage2(self):
        filePath, _ = QtWidgets.QFileDialog.getOpenFileName(None, 'Open File', '', "Image file(*.jpg *.png)")
        _, fileName = os.path.split(filePath)
        self.ui.label_image2.setText(fileName)
        self.filename2 = fileName
        self.img2 = cv2.imread(filePath)
    
    def drawContour(self):
        resized = cv2.resize(self.img1, (0, 0), fx=0.5, fy=0.5)
        gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
        ret, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
        image = cv2.GaussianBlur(binary, (7, 7), 0)
        edged = cv2.Canny(image, 30, 200)
        self.contours1, _ = cv2.findContours(edged, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
        contour1 = cv2.drawContours(resized, self.contours1, -1, (255, 255, 0), 1)
        cv2.imshow('Contours 1', contour1)

        resized = cv2.resize(self.img2, (0, 0), fx=0.5, fy=0.5)
        gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
        ret, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
        image = cv2.GaussianBlur(binary, (7, 7), 0)
        edged = cv2.Canny(image, 30, 200)
        self.contours2, _ = cv2.findContours(edged, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
        contour2 = cv2.drawContours(resized, self.contours2, -1, (255, 255, 0), 1)
        cv2.imshow('Contours 2', contour2)
    
    def countRings(self):
        self.ui.label_ring1.setText("There are " + str(int(len(self.contours1)/2)) + " rings in " + self.filename1)
        self.ui.label_ring2.setText("There are " + str(int(len(self.contours2)/2)) + " rings in " + self.filename2)

    def findCorners(self):
        objp=np.zeros((11*8, 3), np.float32)
        objp[:, :2]=np.mgrid[0:11, 0:8].T.reshape(-1, 2)
        self.objpoints=[]
        self.imgpoints=[]
        for file in os.listdir(self.folderpath):
            f = os.path.join(self.folderpath, file)
            image = cv2.imread(f)
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            self.img_size = gray.shape[::-1]
            self.h, self.w = image.shape[:2]
            ret, corners = cv2.findChessboardCorners(image, (11, 8))
            if ret:
                self.objpoints.append(objp)
                self.imgpoints.append(corners)
            cv2.drawChessboardCorners(image, (11, 8), corners, ret)
            image = cv2.resize(image, (0, 0), fx=0.5, fy=0.5)
            cv2.imshow('Corners', image)
            cv2.waitKey(500)
        cv2.destroyAllWindows()
    
    def findIntrinsic(self):
        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(self.objpoints, self.imgpoints, self.img_size, None, None)
        print("Intrinsic:")
        print(mtx)

    def findExtrinsic(self):
        self.ui.comboBox
        idx = int(self.ui.comboBox.currentText())
        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(self.objpoints, self.imgpoints, self.img_size, None, None)
        rotation_matrix = cv2.Rodrigues(rvecs[idx-1], jacobian=0)
        rotation_matrix = np.array(rotation_matrix[0])
        Extrinsic = np.concatenate((rotation_matrix,tvecs[idx-1]), axis = 1)
        print("Extrinsic:")
        print(Extrinsic)
    
    def findDistortion(self):
        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(self.objpoints, self.imgpoints, self.img_size, None, None)
        print("Distortion:")
        print(dist)

    def undistort(self):
        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(self.objpoints, self.imgpoints, self.img_size, None, None)
        for file in os.listdir(self.folderpath):
            f = os.path.join(self.folderpath, file)
            image = cv2.imread(f)         
            image = cv2.resize(image, (0, 0), fx = 0.5, fy = 0.5, interpolation=cv2.INTER_AREA)
            original = cv2.imread(f)         
            original = cv2.resize(original, (1007,1007), interpolation=cv2.INTER_AREA)
            h,  w = image.shape[:2]
            newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w,h), 1, (w,h))
            dst = cv2.undistort(image, mtx, dist, None, newcameramtx)
            x, y, w, h = roi
            dst = dst[y:y+h, x:x+w]
            result = np.concatenate((original, dst), axis=1)
            cv2.imshow('Undistorted', result)
            cv2.waitKey(500)
        cv2.destroyAllWindows()

    def showWordsOnBoard(self):
        text = self.ui.plainTextEdit.toPlainText()
        text = text.upper()
        for i in range(1,6):
            x = 7
            y = 5
            objpoints = []
            imgpoints = []
            filename = self.folderpath + "/" + str(i) + ".bmp"
            image = cv2.imread(filename)
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            ret, corners = cv2.findChessboardCorners(gray, (11, 8), None)
            objp = np.zeros((11 * 8, 3), np.float32)
            objp[:, :2] = np.mgrid[0:11, 0:8].T.reshape(-1, 2)
            if ret == True:
                objpoints.append(objp)
                imgpoints.append(corners)
            numchar = 1
            for char in text:
                fs = cv2.FileStorage(self.folderpath + '/Q2_lib/alphabet_lib_onboard.txt', cv2.FILE_STORAGE_READ)
                ch = fs.getNode(char).mat() 
                ch = ch.reshape(-1, 3)
            
                for j in range(ch.shape[0]):
                    ch[j][0] += x
                    ch[j][1] += y
                pyramids = np.array(ch, dtype = np.float32)
                ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
                twodpro, jac = cv2.projectPoints(pyramids, rvecs[0], tvecs[0], mtx, dist)
                for j in range(0, ch.shape[0], 2):
                    image = cv2.line(image, (int(twodpro[j][0][0]), int(twodpro[j][0][1])), (int(twodpro[j+1][0][0]), int(twodpro[j+1][0][1])), (0, 0, 255), 10)
                x -= 3
                numchar += 1
                if numchar == 4:
                    x = 7
                    y = 2
            image = cv2.resize(image, (0, 0), fx=0.5, fy=0.5)
            cv2.imshow('img', image)
            cv2.waitKey(1000)
        cv2.destroyAllWindows()

    def showWordsVertically(self):
        text = self.ui.plainTextEdit.toPlainText()
        text = text.upper()
        for i in range(1,6):
            x = 7
            y = 5
            objpoints = []
            imgpoints = []
            filename = self.folderpath + "/" + str(i) + ".bmp"
            image = cv2.imread(filename)
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            ret, corners = cv2.findChessboardCorners(gray, (11, 8), None)
            objp = np.zeros((11 * 8, 3), np.float32)
            objp[:, :2] = np.mgrid[0:11, 0:8].T.reshape(-1, 2)
            if ret == True:
                objpoints.append(objp)
                imgpoints.append(corners)
            numchar = 1
            for char in text:
                fs = cv2.FileStorage(self.folderpath + '/Q2_lib/alphabet_lib_vertical.txt', cv2.FILE_STORAGE_READ)
                ch = fs.getNode(char).mat() 
                ch = ch.reshape(-1, 3)
            
                for j in range(ch.shape[0]):
                    ch[j][0] += x
                    ch[j][1] += y
                pyramids = np.array(ch, dtype = np.float32)
                ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
                twodpro, jac = cv2.projectPoints(pyramids, rvecs[0], tvecs[0], mtx, dist)
                for j in range(0, ch.shape[0], 2):
                    image = cv2.line(image, (int(twodpro[j][0][0]), int(twodpro[j][0][1])), (int(twodpro[j+1][0][0]), int(twodpro[j+1][0][1])), (0, 0, 255), 10)
                x -= 3
                numchar += 1
                if numchar == 4:
                    x = 7
                    y = 2
            image = cv2.resize(image, (0, 0), fx=0.5, fy=0.5)
            cv2.imshow('img', image)
            cv2.waitKey(1000)
        cv2.destroyAllWindows()

    def drawMatch(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            x = x * 4
            y = y * 4
            if x > self.disparity.shape[1]:
                return
            disp = int(self.disparity[y][x] / 16)
            if disp <= 0:
                return
                
            imgRcopy = self.img2.copy()
            point = (x - disp, y) 
            imgRcopy = cv2.circle(imgRcopy, point, 10, (0, 255, 0), -1)
            cv2.imshow('imgR', cv2.resize(imgRcopy, (0, 0), fx=0.25, fy=0.25))

    def stereoDisparityMap(self):
        imgL = self.img1
        imgR = self.img2
        grayL = cv2.cvtColor(imgL, cv2.COLOR_BGR2GRAY)
        grayR = cv2.cvtColor(imgR, cv2.COLOR_BGR2GRAY)
        stereo = cv2.StereoBM_create(numDisparities=256, blockSize=25)
        self.disparity = stereo.compute(grayL, grayR)
        disparity_norm = cv2.normalize(
            self.disparity,
            self.disparity,
            alpha=0,
            beta=255,
            norm_type=cv2.NORM_MINMAX,
            dtype=cv2.CV_8U,
        )
        cv2.imshow('imgL', cv2.resize(imgL, (0, 0), fx=0.25, fy=0.25))
        cv2.imshow('imgR', cv2.resize(imgR, (0, 0), fx=0.25, fy=0.25))
        cv2.imshow('disparity', cv2.resize(disparity_norm, (0, 0), fx=0.25, fy=0.25))
       
        cv2.setMouseCallback('imgL', self.drawMatch)


