import cv2
import os
import sys
import numpy as np
from datetime import datetime
from PyQt5 import QtGui, QtCore
from PyQt5.QtWidgets import QDialog, QApplication, QMainWindow, QMessageBox
from PyQt5.uic import loadUi
import random

# class USER(QDialog):        # Dialog box for entering name and key of new dataset.
#     """USER Dialog """
#     def __init__(self):
#         super(USER, self).__init__()
#         loadUi("user_info.ui", self)

#     def get_name_key(self):
#         name = self.name_label.text()
#         key = int(self.key_label.text())
#         return name, key

class DL(QMainWindow):        # Main application 
    """Main Class"""
    def __init__(self):
        super(DL, self).__init__()
        loadUi("main.ui", self)
        
        # Initializer
        self.timer = QtCore.QTimer()
        self.stop_btn.setEnabled(False)
        self.camera_id = 0
        self.image = cv2.imread("icon/yolov3.jpg", 1)
        self.rcnn_img = self.image.copy()
        self.yolo_img = self.image.copy()
        # Menu
        self.menubar.addAction("About").triggered.connect(self.about_info)
        self.menubar.addAction("Help").triggered.connect(self.help_info)
        # Actions 
        
        self.exit_btn.clicked.connect(self.close)
        self.start_btn.clicked.connect(self.start_detection)
        self.stop_btn.clicked.connect(self.stop_detection)
        self.open_btn.clicked.connect(self.open_file)
        self.recorder.toggled.connect(self.record_video)
        self.downloader.clicked.connect(self.download_image)

        self.open_btn.setEnabled(False)
        # Radio
        self.image_file.clicked.connect(self.img_set)
        self.video_file.clicked.connect(self.video_set)
        self.live_camera.clicked.connect(self.camera_set)
        




    # helper function

    def close(self):
        sys.exit()

    def start_detection(self):
        self.start_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self.start_timer()
        print("Detection started")

    def stop_detection(self):
        if self.timer.isActive():
            self.stop_btn.setEnabled(False)
            self.start_btn.setEnabled(True)
            self.stop_timer()
        print("stopped")

    def open_file(self):
        
        print("open file started")

    def img_set(self):
        self.open_btn.setEnabled(True)
        if self.timer.isActive():
            self.start_btn.setEnabled(True)
            self.stop_btn.setEnabled(False)
        print("Img started")

    def video_set(self):
        self.camera_id = "2h.mp4"
        self.open_btn.setEnabled(True)
        if self.timer.isActive():
            self.start_btn.setEnabled(True)
            self.stop_btn.setEnabled(False)
            self.stop_timer()
        print("video")

    def camera_set(self):
        self.camera_id = 0 # re-initialize camera
        self.open_btn.setEnabled(False)
        if self.timer.isActive():
            self.stop_btn.setEnabled(True)
            self.stop_timer()

        print("camera_set file started")

    def record_video(self):
        if self.recorder.isChecked() and self.timer.isActive():
            self.recorder.setText("Stop")
            # save both video
            try:
                fourcc = cv2.VideoWriter_fourcc(*'XVID')
                output_file_name = str(random.randrange(1,101))
                yolo = output_file_name +'-yolo.avi'
                rcnn = output_file_name +'-rcnn.avi'
                path = os.path.join(os.getcwd(),"recordings")
                os.makedirs(path, exist_ok=True)
                self.video_output_yolo = cv2.VideoWriter(os.path.join(path,yolo),fourcc, 20.0, (640,480))
                self.video_output_rcnn = cv2.VideoWriter(os.path.join(path,rcnn),fourcc, 20.0, (640,480), False)
            except Exception as e:
                QMessageBox().about(self, "Information",str(e))

        elif not self.recorder.isChecked():
            self.recorder.setText("Record Video")
        else:
            QMessageBox().about(self, "Information", "Video can be recorded when capture is ON!")
            self.recorder.setChecked(False)
    
    def download_image(self):
          # def save_image(self):       # Save image captured using the save button.
        location = "pictures"
        name = str(random.randrange(1,101))
        file_name_yolo = name+"_yolo.jpg"
        file_name_rcnn = name+"_rcnn.jpg"
        os.makedirs(os.path.join(os.getcwd(),location), exist_ok=True)
        cv2.imwrite(os.path.join(os.getcwd(),location,file_name_rcnn), self.rcnn_img)
        cv2.imwrite(os.path.join(os.getcwd(),location,file_name_yolo), self.yolo_img)
        

    def start_timer(self):      # start the timeer for execution.
        self.capture = cv2.VideoCapture(self.camera_id)
        self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.timer.timeout.connect(self.update_image)
        self.timer.start(5)

    def stop_timer(self):       # stop timer or come out of the loop.
        self.timer.stop()
        self.ret = False
        self.capture.release()
        
    def update_image(self):     # update canvas every time according to time set in the timer.
        self.ret, self.image = self.capture.read()
        self.image = cv2.flip(self.image, 1)
        self.yolo_img = self.image
        self.rcnn_img = cv2.Canny(self.yolo_img, 50, 100)
        #faces = self.get_faces()
        #self.draw_rectangle(faces)
        if self.recorder.isChecked():
            self.video_output_yolo.write(self.yolo_img)
            self.video_output_rcnn.write(self.rcnn_img)
        self.display()

    def display(self):      # Display in the canvas, video feed.
        yolo_img = self.pix_image(self.yolo_img)
        rcnn_img = self.pix_image(self.rcnn_img)
        self.faster_rcnn.setPixmap(QtGui.QPixmap.fromImage(rcnn_img))
        self.yolo_v3.setPixmap(QtGui.QPixmap.fromImage(yolo_img))
        self.faster_rcnn.setScaledContents(True)
        self.yolo_v3.setScaledContents(True)

    def pix_image(self, image): # Converting image from OpenCv to PyQT compatible image.
        qformat = QtGui.QImage.Format_RGB888  # only RGB Image
        if len(image.shape) >= 3:
            r, c, ch = image.shape
        else:
            r, c = image.shape
            qformat = QtGui.QImage.Format_Indexed8
        pixImage = QtGui.QImage(image, c, r, image.strides[0], qformat)
        return pixImage.rgbSwapped()

    
    
    # def absolute_path_generator(self):      # Generate all path in dataset folder.
    #     separator = "-"
    #     for folder, folders, _ in os.walk(os.path.join(os.getcwd(),"datasets")):
    #         for subfolder in folders:
    #             subject_path = os.path.join(folder,subfolder)
    #             key, _ = subfolder.split(separator)
    #             for image in os.listdir(subject_path):
    #                 absolute_path = os.path.join(subject_path, image)
    #                 yield absolute_path,key

    
    # def get_gray_image(self):       # Convert BGR image to GRAY image.
    #     return cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)

    

    # def draw_rectangle(self, faces):        # Draw rectangle either in face, eyes or smile.
    #     for (x, y, w, h) in faces:
    #         roi_gray_original = self.get_gray_image()[y:y + h, x:x + w]
    #         roi_gray = self.resize_image(roi_gray_original, 92, 112)
    #         roi_color = self.image[y:y+h, x:x+w]
    #         if self.recognize_face_btn.isChecked():
    #             try:
    #                 predicted, confidence = self.face_recognizer.predict(roi_gray)
    #                 name = self.get_all_key_name_pairs().get(str(predicted))
    #                 self.draw_text("Recognizing using: "+self.algorithm, 70,50)
    #                 if self.lbph_algo_radio.isChecked():
    #                     if confidence > 105:
    #                         msg = "More like [" + name + "]"
    #                     else:
    #                         confidence = "{:.2f}".format(100 - confidence)
    #                         msg = name
    #                     self.progress_bar_recognize.setValue(float(confidence))
    #                 else:
    #                     msg = name
    #                     self.progress_bar_recognize.setValue(int(confidence%100))
    #                     confidence = "{:.2f}".format(confidence)

    #                 self.draw_text(msg, x-5,y-5)
    #             except Exception as e:
    #                 self.print_custom_error("Unable to Pridict due to")
    #                 print(e)

    #         if self.eye_rect_radio.isChecked():     # If eye radio button is checked.
    #             eyes = self.get_eyes(roi_gray_original)
    #             for (ex, ey, ew, eh) in eyes:
    #                 cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2)
    #         elif self.smile_rect_radio.isChecked():     # If smile radio button is checked.
    #             smiles = self.get_smiles(roi_gray_original)
    #             for (sx, sy, sw, sh) in smiles:
    #                 cv2.rectangle(roi_color, (sx, sy), (sx+sw, sy+sh), (0, 255, 0), 2)
    #         else:       # If face radio button is checked.
    #             cv2.rectangle(self.image, (x, y), (x + w, y + h), (0, 255, 0), 2)
            
    # def time(self):     # Get current time.
    #     return datetime.now().strftime("%d-%b-%Y:%I-%M-%S")

    # def draw_text(self, text, x=20, y=20, font_size=2, color = (0, 255, 0)): # Draw text in current image in particular color.
    #     cv2.putText(self.image, text, (x,y), cv2.FONT_HERSHEY_PLAIN, 1.6, color, font_size)

    # def resize_image(self, image, width=280, height=280): # Resize image before storing.
    #     return cv2.resize(image, (width,height), interpolation = cv2.INTER_CUBIC)

    # def print_custom_error(self, msg):      # Print custom error message/
    #     print("="*100)
    #     print(msg)
    #     print("="*100)
    
    # # Main Menu
    
    def about_info(self):       # Menu Information of info button of application.
        msg_box = QMessageBox()
        msg_box.setText('''
            AUFR (authenticate using face recognition) is an Python/OpenCv based
            face recognition application. It uses Machine Learning to train the
            model generated using haar classifier.
            Eigenfaces, Fisherfaces and LBPH algorithms are implemented.
            The code of this application is available at github @indian-coder.
        ''')
        msg_box.setInformativeText('''
            Ambedkar Institute of Technology, NCT of Delhi-110031.
            Mentor: Dr. Aatri Jain
            Team  : Md. Danish, Sumit Chaurasia
            September, 30th, 2018
            ''')
        msg_box.setWindowTitle("About AUFR")
        msg_box.exec_()

    def help_info(self):       # Menu Information of help button of application.
        msg_box = QMessageBox()
        msg_box.setText('''
            This application is capable of creating datasets, generating models,
            recording videos and clicking images.
            Detection of face, eyes, smile are also implemented.
            Recognition of person is primary job of this application.
        ''')
        msg_box.setInformativeText('''
            Follow these steps to use this application
            1. Generate atlest two datasets.
            2. Train all algoritmic model using given radio buttons.
            3. Recognize person.
            ''')
        msg_box.setWindowTitle("Help")
        msg_box.exec_()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    ui = DL()         # Running application loop.
    ui.show()
    sys.exit(app.exec_())       #  Exit application.