import sys
import os
import face_alignment
import numpy as np
import imghdr
import dlib
import cv2
from skimage import io
from PyQt5.QtGui import *
from PyQt5.Qt import *
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
import qdarkgraystyle

#contains name, points, pixmap image, paths, and landmarkpaths of image
class ImageSet:
    def __init__(self):
        self.__pixmslap = None
        self.__path = ""
        self.__point = []
        self.__landmarkPath = []
        self.__name = ""

    @property  # get pixmap
    def pixmap(self):
        return self.__pixmap

    @pixmap.setter  # set pixmap
    def pixmap(self, pixmap):
        self.__pixmap = pixmap

    @property  # get path
    def path(self):
        return self.__path

    @path.setter  # set path
    def path(self, path):
        self.__path = path

    @property  # get point
    def point(self):
        return self.__point

    @point.setter  # set point
    def point(self, point):
        self.__point = point

    @property  # get landmarkPath
    def landmarkPath(self):
        return self.__landmarkPath

    @landmarkPath.setter  # set landmarkPath
    def landmarkPath(self, landmarkPath):
        self.__landmarkPath = landmarkPath

    @property  # get name
    def name(self):
        return self.__name

    @name.setter  # set name
    def name(self, name):
        self.__name = name


#upload the image to viewer and add landmarks on it
class PhotoViewer(QGraphicsView):

    def __init__(self):
        super(PhotoViewer, self).__init__()
        self._zoom = 0  # size of zoom
        self._empty = True  # whether viwer is empty
        self._scene = QGraphicsScene(self)  # scene to be uploaded
        self._photo = QGraphicsPixmapItem()  # photo that goes into scene
        self._scene.addItem(self._photo)  # add photo into scene
        self.setScene(self._scene)  # set scene into viwer
        self.pixmap = QPixmap()
        self.realPixmap = QPixmap()
        self.highReso = False #whether the image set is high resolution or not.

        self.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOn)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOn)

    def hasPhoto(self):
        return not self._empty

    def fitInView2(self):
        rect = QRectF(self._photo.pixmap().rect())
        if not rect.isNull():
            self._scene.setSceneRect(rect)

            if self.hasPhoto():
                unity = self.transform().mapRect(QRectF(0, 0, 1, 1))
                self.scale(1 / unity.width(), 1 / unity.height())

                if self.highReso:
                    self.fitInView(rect, Qt.KeepAspectRatio)

            self._zoom = 0

    def setPhoto(self, pixmap=None):
        self._zoom = 0

        if pixmap and not pixmap.isNull():

            dw = QDesktopWidget()
            dwWidth = dw.width()
            dwHeight = dw.height()

            pixmapWidth = pixmap.width()
            pixmapHeight = pixmap.height()

            self.realPixmap = pixmap

            if (pixmapWidth > dwWidth) or (pixmapHeight > dwHeight):
                self.pixmap = pixmap.scaled(dw.width(), dw.height(), Qt.KeepAspectRatio)
                self.highReso = True
            else:
                self.highReso = False

            self._empty = False
            self._photo.setPixmap(pixmap)

        else:
            self._empty = True
            self._photo.setPixmap(QPixmap())

        self.fitInView2()

    def keyPressEvent(self, event):
        if event.key() == Qt.Key_Control:
            self.setDragMode(QGraphicsView.ScrollHandDrag)

    def keyReleaseEvent(self, event):
        if event.key() == Qt.Key_Control:
            self.setDragMode(QGraphicsView.NoDrag)

    def wheelEvent(self, event):
        if self.hasPhoto():
            modifiers = QApplication.keyboardModifiers()
            if modifiers == Qt.ControlModifier:  # if ctrl key is pressed
                if event.angleDelta().y() > 0:  # if scroll wheel forward
                    factor = 1.25
                    self._zoom += 1
                else:
                    factor = 0.8  # if scroll wheel backward
                    self._zoom -= 1
                if self._zoom > 0:
                    self.scale(factor, factor)
                elif self._zoom == 0:
                    self.fitInView2()
                else:
                    self._zoom = 0

    def addItem(self, item):
        return self._scene.addItem(item)


class Landmark_path(QGraphicsPathItem):

    def __init__(self, path):
        super(Landmark_path, self).__init__()
        self.path = path
        self.setPath(path)
        pos = self.scenePos()
        self.x = pos.x()
        self.y = pos.y()
        self.setFlag(QGraphicsItem.ItemIsMovable, True)
        self.setFlag(QGraphicsItem.ItemIsSelectable, True)

    def mousePressEvent(self, event):
        event.accept()
        super(Landmark_path, self).mousePressEvent(event)

    def mouseMoveEvent(self, event):
        event.accept()
        super(Landmark_path, self).mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        event.accept()
        super(Landmark_path, self).mouseReleaseEvent(event)
        pos = self.scenePos()
        point = self.mapToScene(pos.x(), pos.y())
        self.x = point.x()
        self.y = point.y()

    def returnCoordinates(self):
        self.x = self.scenePos().x()
        self.y = self.scenePos().y()
        return np.array([self.x, self.y])


class MainWindow(QMainWindow):

    def __init__(self):
        super().__init__()
        self.title = 'face landmark detection Program'
        self.dw = QDesktopWidget()  # fit to the size of desktop
        self.x = 0
        self.y = 0
        self.width = self.dw.width()  # width of the desktop
        self.height = self.dw.height()  # height of the desktop
        self._label = QLabel()  # labels that show no photo uploaded warnings

        self.currentImage = ImageSet()  # current image
        self.imageFolder = []  # folder of images
        self.imageFolderIndex = 0  # index of current image shown in image folder
        self.imageFolderText = {}  # dictionary of texts that correspond to image in folder
        self.drawn = False #whether landmarks are drawn or not

        self.initUI()  # initiate UI

    def initUI(self):

        # resize the size of window
        self.setWindowTitle(self.title)
        self.setGeometry(self.x, self.y, self.width, self.height)

        # add graphic View into centralWidget
        self.viewer = PhotoViewer()
        self.setCentralWidget(self.viewer)
        #self.viewer.setStyleSheet("background-color:#3C3530")

        # create Widget and layout for buttons on the right
        buttonWidget = QWidget()

        #create radioButton for detector
        groupBox = QGroupBox("Type of Detector", buttonWidget)
        groupBox.move(buttonWidget.x() + 5, buttonWidget.y() + 10)
        groupBox.resize(140,120)
        groupBox.setStyleSheet("background-color:#F16B6F")

        self.radio1 = QRadioButton("Pytorch", groupBox)
        self.radio1.move(buttonWidget.x() + 5, buttonWidget.y() + 25)
        self.radio1.clicked.connect(self.radioButtonClicked)
        self.radio1.setChecked(True) #default is pytorch

        self.radio2 = QRadioButton("Dlib", groupBox)
        self.radio2.move(buttonWidget.x() + 5, buttonWidget.y() + 45)
        self.radio2.clicked.connect(self.radioButtonClicked)

        self.radio3 = QRadioButton("CLM", groupBox)
        self.radio3.move(buttonWidget.x() + 5, buttonWidget.y() + 65)
        self.radio3.clicked.connect(self.radioButtonClicked)

        self.radio4 = QRadioButton("CSC", groupBox)
        self.radio4.move(buttonWidget.x() + 5, buttonWidget.y() + 85)
        self.radio4.clicked.connect(self.radioButtonClicked)

        # create upload label
        uploadLb = QLabel("1. Upload", buttonWidget)
        uploadLb.setFont(QFont("Book Antiqua", 14, QFont.Bold))
        uploadLb.move(buttonWidget.x(), buttonWidget.y() + 200)

        # create upload buttons
        uploadImBut = QPushButton('Image(ctrl+i)', buttonWidget)
        uploadImBut.setFont(QFont('Futura',13, QFont.Bold))
        uploadImBut.resize(140, 50)
        uploadImBut.move(buttonWidget.x() + 5, buttonWidget.y() + 235)

        uploadTeBut = QPushButton('Text(ctrl+t)', buttonWidget)
        uploadTeBut.setFont(QFont('Futura', 13, QFont.Bold))
        uploadTeBut.resize(140, 50)
        uploadTeBut.move(buttonWidget.x() + 150, buttonWidget.y() + 235)

        uploadFoBut = QPushButton('folder(ctrl+f)', buttonWidget)
        uploadFoBut.setFont(QFont('Futura', 13, QFont.Bold))
        uploadFoBut.resize(140, 50)
        uploadFoBut.move(buttonWidget.x() + 5, buttonWidget.y() + 290)

        # create detect label
        detectLb = QLabel("2. Detect", buttonWidget)
        detectLb.setFont(QFont("Book Antiqua", 14, QFont.Bold))
        detectLb.move(buttonWidget.x(), buttonWidget.y() + 450)

        # create detect buttons
        detectBut = QPushButton('Detect(ctrl+d)', buttonWidget)
        detectBut.setFont(QFont('Futura', 13, QFont.Bold))
        detectBut.resize(140, 50)
        detectBut.move(buttonWidget.x() + 5, buttonWidget.y() + 485)

        detectClBut = QPushButton('Detection Clear(ctrl+c)', buttonWidget)
        detectClBut.setFont(QFont('Futura', 13, QFont.Bold))
        detectClBut.resize(140, 50)
        detectClBut.move(buttonWidget.x() + 150, buttonWidget.y() + 485)

        # create save label
        saveLb = QLabel("3. Save", buttonWidget)
        saveLb.setFont(QFont("Book Antiqua", 13, QFont.Bold))
        saveLb.move(buttonWidget.x(), buttonWidget.y() + 600)

        # create save buttons
        saveBut = QPushButton('Save(ctrl+s)', buttonWidget)
        saveBut.setFont(QFont('Futura', 13, QFont.Bold))
        saveBut.resize(140, 50)
        saveBut.move(buttonWidget.x() + 5, buttonWidget.y() + 635)

        # create left and right buttons
        right_arrow_pixmap = QPixmap('Icons/right_arrow.png')
        left_arrow_pixmap = QPixmap('Icons/left_arrow.png')
        right_arrow_icon = QIcon(right_arrow_pixmap)
        left_arrow_icon = QIcon(left_arrow_pixmap)

        rightArrowBut = QPushButton(buttonWidget)
        rightArrowBut.setIcon(right_arrow_icon)
        leftArrowBut = QPushButton(buttonWidget)
        leftArrowBut.setIcon(left_arrow_icon)

        rightArrowBut.move(buttonWidget.x() + 150, buttonWidget.y() + 800)
        leftArrowBut.move(buttonWidget.x() + 120, buttonWidget.y() + 800)

        # create text label
        self.textLb = QLabel(buttonWidget)
        self.textLb.move(buttonWidget.x() + 5, buttonWidget.y() + 380)
        self.textLb.setFont(QFont('Futura', 12))
        self.textLb.setText('Image: ')

        # create qdockwidget and add the button widget to it
        self.qDockWidget = QDockWidget("")
        self.qDockWidget.setWidget(buttonWidget)
        self.qDockWidget.setFloating(False)
        self.qDockWidget.setFeatures(QDockWidget.NoDockWidgetFeatures)
        self.addDockWidget(Qt.RightDockWidgetArea, self.qDockWidget)
        self.qDockWidget.setFixedSize(300, self.dw.height())

        # upload button connection
        uploadImBut.clicked.connect(self.uploadImButClicked)
        uploadImBut.setShortcut("ctrl+i")
        uploadTeBut.clicked.connect(self.uploadTeButClicked)
        uploadTeBut.setShortcut("ctrl+t")
        uploadFoBut.clicked.connect(self.uploadFoButClicked)
        uploadFoBut.setShortcut("ctrl+f")

        # detection button connection
        detectBut.clicked.connect(self.detectButClicked)
        detectBut.setShortcut("ctrl+d")
        detectClBut.clicked.connect(self.detectClButClicked)
        detectClBut.setShortcut("ctrl+c")

        # save button connectoin
        saveBut.clicked.connect(self.saveButClicked)
        saveBut.setShortcut("ctrl+s")

        # right and left arrow button connection
        rightArrowBut.clicked.connect(self.rightArrowButClicked)
        rightArrowBut.setShortcut(Qt.Key_Right)
        leftArrowBut.clicked.connect(self.leftArrowButClicked)
        leftArrowBut.setShortcut(Qt.Key_Left)

    def drawPoints(self, points):

        # create Painterpath with text and circle and add them to the viewer

        if len(self.currentImage.landmarkPath) != 0:
            self.detectClButClicked()
            self.currentImage.landmarkPath.clear()

        index = 1

        pixmapWidth = self.currentImage.pixmap.width()
        pixmapHeight = self.currentImage.pixmap.height()
        pixmapSize = pixmapHeight * pixmapWidth

        EllipSize = pixmapSize / 1000000

        for x, y in points:

            if index > 68:
                index = 1

            path = QPainterPath()
            font = QFont('Times', 2)  # font
            font.setPointSize(EllipSize + 2)
            font.setWeight(0.1)  # make font thinner
            font.setLetterSpacing(QFont.PercentageSpacing, 150)
            path.addText(5, 5, font, str(index))

            path.addEllipse(0, 0, EllipSize, EllipSize)
            qPen = QPen()

            # set color for each landmarkPath
            if index <= 17:
                qPen.setColor(QColor(150, 0, 0))
            elif 18 <= index and index <= 26:
                qPen.setColor(QColor(0, 150, 0))
            elif 28 <= index and index <= 35:
                qPen.setColor(QColor(0, 0, 150))
            elif 37 <= index and index <= 47:
                qPen.setColor(QColor(100, 100, 100))
            elif 49 <= index and index <= 68:
                qPen.setColor(QColor(50, 50, 50))

            # create landmark_point and add them to viwer
            landmark_path = Landmark_path(path)
            landmark_path.setPos(x, y)
            landmark_path.setPen(qPen)

            self.currentImage.landmarkPath.append(landmark_path)
            self.viewer.addItem(landmark_path)

            index += 1

        self.drawn = True

    def rightArrowButClicked(self):

        # if imageFolder empty, rightArrowBut does not operate
        if len(self.imageFolder) == 0:
            pass
        else:  # if imageFolder is not empty

            # if there is landmark path on scene
            if self.drawn:
                self.detectClButClicked()

            if self.imageFolderIndex >= len(self.imageFolder) - 1:
                self.imageFolderIndex = 0
            else:
                self.imageFolderIndex += 1

            # get corresponding image from folder
            self.currentImage = self.imageFolder[self.imageFolderIndex]
            self.viewer.setPhoto(self.currentImage.pixmap)

            # if there is text file with points to the image
            if self.currentImage.name in self.imageFolderText:

                numbers = []

                for line in self.imageFolderText[self.currentImage.name]:
                    numbers.append(line.split())

                for j in range(0, 2):
                    for i in range(0, 68):
                        numbers[i][j] = float(numbers[i][j])

                self.drawPoints(numbers)

            self.textLb.setText('Image :' + str(self.currentImage.name))
            self.textLb.adjustSize()

    def leftArrowButClicked(self):

        # if imageFolder empty, rightArrowBut does not operate
        if len(self.imageFolder) == 0:
            pass
        else:  # if imageFolder is not empty

            # if there is landmark path on scene
            if self.drawn:
                self.detectClButClicked()

            if self.imageFolderIndex <= 0:
                self.imageFolderIndex = len(self.imageFolder) - 1
            else:
                self.imageFolderIndex -= 1

            # get corresponding image from folder
            self.currentImage = self.imageFolder[self.imageFolderIndex]
            self.viewer.setPhoto(self.currentImage.pixmap)

            # if there is text file with points to the image
            if self.currentImage.name in self.imageFolderText:

                numbers = []

                for line in self.imageFolderText[self.currentImage.name]:
                    numbers.append(line.split())

                for j in range(0, 2):
                    for i in range(0, 68):
                        numbers[i][j] = float(numbers[i][j])

                self.drawPoints(numbers)

            self.textLb.setText('Image :' + str(self.currentImage.name))
            self.textLb.adjustSize()

    def saveButClicked(self):
        ##
        if self.currentImage.landmarkPath != []:
        # get the path of directory where image is located at

            filepath_full = os.path.splitext(self.currentImage.path)[0] + '.txt'

            f = open(filepath_full, 'w')
            temp = []

            for point in self.currentImage.landmarkPath:
                f.write(str(point.returnCoordinates()).replace('[', '').replace(']', '') + '\n')
                temp.append(str(point.returnCoordinates()).replace('[', '').replace(']', '') + '\n')

            if len(self.imageFolder) != 0: #if there is folder
                if self.currentImage.name in self.imageFolderText: #if there is already corresponding text file, save the changes
                    self.imageFolderText[self.currentImage.name] = temp
                else: #if there is no corresponding text file, add it
                    self.imageFolderText[self.currentImage.name] = temp
            f.close()
        else:
            self.clickMethod()

    def uploadImButClicked(self):
        # if imageFolder is not empty, empty it
        if len(self.imageFolder) != 0:
            self.imageFolder.clear()

        # get the path of file
        fNamePath = QFileDialog.getOpenFileName(self, "Open Image", "Image Files (*.png *.jpg *.bmp *.jpeg *.gif)")
        #fNamePath = QFileDialog.getOpenFileName(self, 'Open file', '/home')
        self.currentImage.path = fNamePath[0]

        if self.currentImage.path == "": #if esc was pressed or no image was chosen, do not set empty path as current image path
            pass
        else:
            # upload images on grpahicView
            self.currentImage.pixmap = QPixmap(self.currentImage.path)
            self.viewer.setPhoto(self.currentImage.pixmap)
            self.textLb.setText('Image :' + str(os.path.basename(self.currentImage.path)))
            self.textLb.adjustSize()

    def uploadFoButClicked(self):
        # get the path of selected directory
        dir_ = QFileDialog.getExistingDirectory(None, 'Open folder:', 'QDir.homePath()', QFileDialog.ShowDirsOnly)

        # if not choose any directory, just pass
        if not dir_:
            pass
        else:
            # if imageFolder is not empty, empty it
            if len(self.imageFolder) != 0:
                self.imageFolder.clear()

            # put each corresponding image into imageFolder
            for files_ext in sorted(os.listdir(dir_)):

                imagePath = dir_ + '/' + files_ext
                self.textLb.setText('Image : ' + str(files_ext))
                self.textLb.adjustSize()

                try:
                    if imghdr.what(imagePath) is "jpg" or imghdr.what(imagePath) is "png" or imghdr.what(imagePath) is "jpeg" or imghdr.what(imagePath) is "gif":
                        imagePixmap = QPixmap(imagePath)
                        image = ImageSet()
                        image.pixmap = imagePixmap
                        image.path = imagePath
                        image.name = files_ext

                        #self.nameLb.setText(image.name)
                        #self.nameLb.adjustSize()

                        self.imageFolder.append(image)
                    elif files_ext.endswith(".txt"):
                        imagePath = dir_ + '/' + files_ext
                        f = open(imagePath, 'r')

                        points = f.readlines()

                        name = files_ext.replace(".txt", ".jpg")

                        self.imageFolderText[name] = points
                    else:
                        pass
                except IsADirectoryError: #if there is a directory inside folder
                    self.clickMethod4()
                    pass
                except PermissionError:#if there is permission error with the file
                    self.clickMethod5()
                    pass

            self.imageFolderIndex = 0

            if len(self.imageFolder) == 0: #if there is no file in imageFolder
                pass
            else:
                self.currentImage = self.imageFolder[self.imageFolderIndex]
                self.viewer.setPhoto(self.currentImage.pixmap)

                # if there is text file with points to the image
                if self.currentImage.name in self.imageFolderText:

                    numbers = []

                    for line in self.imageFolderText[self.currentImage.name]:
                        numbers.append(line.split())

                    for j in range(0, 2):
                        for i in range(0, 68):
                            numbers[i][j] = float(numbers[i][j])

                    self.drawPoints(numbers)


    def uploadTeButClicked(self):

        if not self.viewer.hasPhoto():
            self.clickMethod3()
        else:
            fNamePath = QFileDialog.getOpenFileName(self, "Open text", "/home/", "Text Files (*.txt)")

            # get coordinates from text files
            textPath = fNamePath[0]
            filepath_full = os.path.splitext(textPath)[0] + '.txt'

            with open(filepath_full) as data:
                lines = data.readlines()

            numbers = []

            lineNum = 0

            for line in lines:
                numbers.append(line.split())
                lineNum += 1

            if lineNum is 68:

                for j in range(0, 2):
                    for i in range(0, 68):
                        numbers[i][j] = float(numbers[i][j])

                self.drawPoints(numbers)
            else:
                self.clickMethod2()

    def label_update(self):
        update = self.currentImage.path
        #self.textLb.setText(update)
        #self.textLb.setText("")

    def clickMethod(self):
        QMessageBox.about(self, "Warning", "Landmark is empty. Try detection again.")

    def clickMethod2(self):
        QMessageBox.about(self, "Error", " Check text file(*.txt). Some landmarks are missing.")

    def clickMethod3(self):
        QMessageBox.about(self, "Error", "No photo uploaded")

    def clickMethod4(self):
        QMessageBox.about(self, "Error", "There is folder file inside")

    def clickMethod5(self):
        QMessageBox.about(self, "Error", "Permission Error")

    def detectClButClicked(self):

        if not self.viewer.hasPhoto():
            self.clickMethod3()

        else:
            for landmarkPath in self.currentImage.landmarkPath: #remove landmarkpath in image
                self.viewer.scene().removeItem(landmarkPath)

            self.drawn = False

    def detectButClicked(self):
        if not self.viewer.hasPhoto():
            self.clickMethod3()
        elif self.radio1.isChecked():
            self.pytorch_detector()
        elif self.radio2.isChecked():
            self.dlib_detector()

    #adrain detector
    def pytorch_detector(self):
        fa = face_alignment.FaceAlignment(face_alignment.LandmarksType._2D, device='cpu')

        if not self.viewer.highReso:
            input = io.imread(self.currentImage.path)
            self.currentImage.point = fa.get_landmarks(input)[-1]
        else:
            self.viewer.pixmap.toImage().save('../tempo.jpg')
            input = io.imread('../tempo.jpg')
            os.remove('../tempo.jpg')
            self.currentImage.point = fa.get_landmarks(input)[-1]
            xratio = self.viewer.realPixmap.width() / self.viewer.pixmap.width()
            yratio = self.viewer.realPixmap.height() / self.viewer.pixmap.height()

            newPoint = []
            for point in self.currentImage.point:
                newPoint.append(np.array([point[0] * xratio, point[1] * yratio]))
            self.currentImage.point = newPoint

        self.drawPoints(self.currentImage.point)

    #dlib dector
    def dlib_detector(self):
        input = cv2.imread(self.currentImage.path)
        gray = cv2.cvtColor(input, cv2.COLOR_BGR2GRAY)
        detector = dlib.get_frontal_face_detector()
        predictor = dlib.shape_predictor(
            "./shape_predictor_68_face_landmarks.dat"
        )
        dets = detector(gray, 1)
        points_all = []
        for face in dets:
            shape = predictor(input, face)
            for point in shape.parts():
                points = [point.x, point.y]
                points_all.append(points)

        self.currentImage.point = points_all
        self.drawPoints(self.currentImage.point)

    #show which radio button was clicked
    def radioButtonClicked(self):
        msg = ''
        if self.radio1.isChecked():
            msg = 'pytorch'
        elif self.radio2.isChecked():
            msg = 'dlib'
        elif self.radio3.isChecked():
            msg = 'CLM'
        else:
            msg = 'Ensemble of Regression Tree'


if __name__ == '__main__':
    app = QApplication(sys.argv)
    app.setStyleSheet(qdarkgraystyle.load_stylesheet())
    window = MainWindow()
    window.show()
    app.exec()