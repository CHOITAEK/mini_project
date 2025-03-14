import sys
import cv2
import easyocr
import numpy as np
import re
import os
from openvino.runtime import Core
from datetime import datetime
from PyQt5.QtWidgets import QApplication, QWidget, QLabel, QVBoxLayout, QHBoxLayout, QListWidget
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import QThread, pyqtSignal, Qt


class VideoThread(QThread):
    change_pixmap_signal = pyqtSignal(np.ndarray)

    def __init__(self):
        super().__init__()
        self.cap = cv2.VideoCapture(5)  # 카메라 ID 설정
        if not self.cap.isOpened():
            print("카메라를 열 수 없습니다.")
            sys.exit()
        self.running = True

    def run(self):
        while self.running:
            ret, frame = self.cap.read()
            if ret:
                self.change_pixmap_signal.emit(frame)

    def stop(self):
        self.running = False
        self.quit()
        self.wait()
        self.cap.release()


class LicensePlateRecognitionApp(QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()

        # OpenVINO 설정
        self.ie = Core()
        self.plate_model = self.ie.read_model("yolov5s_openvino_model/yolov5s.xml")
        self.plate_compiled_model = self.ie.compile_model(self.plate_model, "CPU")

        self.reader = easyocr.Reader(['ko', 'en'], gpu=False)

        self.thread = VideoThread()
        self.thread.change_pixmap_signal.connect(self.update_image)
        self.thread.start()

        self.captured_image = None

        # 저장 폴더 생성
        if not os.path.exists("image"):
            os.makedirs("image")

    def initUI(self):
        self.setWindowTitle('차량 번호판 인식 시스템')
        self.setGeometry(100, 100, 1280, 720)
        self.setFocusPolicy(Qt.StrongFocus)

        self.label_live = QLabel(self)
        self.label_live.setFixedSize(640, 480)
        self.label_live.setAlignment(Qt.AlignCenter)

        self.label_captured = QLabel(self)
        self.label_captured.setFixedSize(640, 480)
        self.label_captured.setAlignment(Qt.AlignCenter)

        self.list_plates = QListWidget(self)

        layout = QVBoxLayout()
        top_layout = QHBoxLayout()
        bottom_layout = QVBoxLayout()

        left_layout = QVBoxLayout()
        left_layout.setSpacing(5)
        left_layout.addWidget(QLabel("실시간 영상"))
        left_layout.addWidget(self.label_live)

        right_layout = QVBoxLayout()
        right_layout.setSpacing(5)
        right_layout.addWidget(QLabel("분석된 이미지"))
        right_layout.addWidget(self.label_captured)

        top_layout.addLayout(left_layout)
        top_layout.addLayout(right_layout)

        bottom_layout.addWidget(QLabel("인식된 번호판"))
        bottom_layout.addWidget(self.list_plates)

        layout.addLayout(top_layout)
        layout.addLayout(bottom_layout)
        self.setLayout(layout)

    def showEvent(self, event):
        self.grabKeyboard()
        super().showEvent(event)

    def update_image(self, frame):
        self.display_image(frame, self.label_live)
        self.latest_frame = frame.copy()

    def keyPressEvent(self, event):
        if event.key() == Qt.Key_T:
            print("T key pressed")
            if self.latest_frame is not None:
                self.captured_image = self.latest_frame.copy()
                plates, processed_img = self.detect_license_plates(self.captured_image)
                self.display_image(processed_img, self.label_captured)
                self.display_plates(plates)
                self.save_image(processed_img, plates)

    def detect_license_plates(self, img):
        input_tensor = np.expand_dims(img.transpose(2, 0, 1), axis=0).astype(np.float32)
        results = self.plate_compiled_model.infer_new_request({"images": input_tensor})
        detections = results["output"]
        plates = []

        for detection in detections[0]:
            x1, y1, x2, y2, conf, cls = detection[:6]
            if conf > 0.4:
                x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 3)
                plate_img = img[y1:y2, x1:x2]
                for bbox, text, conf in self.reader.readtext(plate_img):
                    if conf > 0.3:
                        filtered_text = re.sub(r'[^가-힣0-9]', '', text)
                        if filtered_text:
                            plates.append(filtered_text)
                            cv2.rectangle(img, tuple(map(int, bbox[0])), tuple(map(int, bbox[2])), (0, 255, 255), 3)
        return plates, img

    def display_image(self, img, label):
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        height, width, channel = img.shape
        bytes_per_line = channel * width
        q_image = QImage(img.data, width, height, bytes_per_line, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(q_image)
        label.setPixmap(pixmap)
        label.setScaledContents(True)

    def display_plates(self, plates):
        self.list_plates.clear()
        for idx, plate in enumerate(plates):
            self.list_plates.addItem(f'차량 {idx+1}: {plate}')

    def save_image(self, img, plates):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"image/captured_{timestamp}.png"
        cv2.imwrite(filename, img)
        print(f"이미지 저장됨: {filename}")
        
        plate_filename = f"image/captured_{timestamp}.txt"
        with open(plate_filename, "w") as f:
            for idx, plate in enumerate(plates):
                f.write(f'차량 {idx+1}: {plate}\n')
        print(f"번호판 저장됨: {plate_filename}")

    def closeEvent(self, event):
        self.thread.stop()
        event.accept()

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = LicensePlateRecognitionApp()
    window.show()
    sys.exit(app.exec_())