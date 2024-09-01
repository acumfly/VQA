import sys
import os
import pandas as pd
import re

from PyQt6 import QtWidgets, QtGui, QtCore
import design


import preprocess_data
import make_preds

# from PIL import ImageTk, Image
# from preprocess_data import preprocess_multitarget, preprocess_binary, preprocess_generation, decode_outputs, preprocess_ques_type
# from make_preds import make_preds_multitarget, make_preds_binary, make_generation, make_preds_ques_type


class VQAApp(QtWidgets.QMainWindow, design.Ui_MainWindow):
    def __init__(self):
        super().__init__()
        self.setupUi(self)
        self.folder_button.clicked.connect(self.browse_folder)
        self.image_path = None
        self.multitarget_test = pd.read_csv('datasets/multitarget_test.csv')
        self.binary_test = pd.read_csv('datasets/binary_test.csv')
        self.multitarget_templates = [
            re.compile(r"^what is present$"),
            re.compile(r"^what does this \w+ show$"),
            re.compile(r"^where is this$"),
            re.compile(r"^where does this belong to$"),
            re.compile(r"^where is this from$"),
            re.compile(r"^where are \w+ \w+ located$"),
            re.compile(r"^where is this part in the figure$")
        ]


        self.folder_button.clicked.connect(self.open_folder)
        self.prediction_button.clicked.connect(self.make_prediction)

        self.current_question = ""
        self.current_image = None

    def open_folder(self):
        options = QtWidgets.QFileDialog.Options()
        options = QtWidgets.QFileDialog.Option.ReadOnly
        file_path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self,
            caption='Выберите изображение',
            directory="",
            filter="Изображения (*.png, *.xmp, *.jpg, *.jpeg, *.bmp, *.gif)",
            options=options
        )
        if file_path:
            self.image_path = file_path
            self.display_image(file_path)

    def display_image(self, path):
        pixmap = QtGui.QPixmap(path)
        if pixmap.isNull():
            QtWidgets.QMessageBox.critical(
                self,
                title="Ошибка",
                text="Не удалось загрузить изображение"
            )
            return

        #масштабируем изображение под размер QLabel
        scaled_pixmap = pixmap.scaled(
            self.image_widget.size(),
            QtCore.Qt.AspectRatioMode.KeepAspectRatio,
            QtCore.Qt.TransformationMode.SmoothTransformation
        )
        self.image_widget.setPixmap(scaled_pixmap)
        self.image_widget.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.current_image = path

    def matches_pattern(self, question, template_list):
        return any(pattern.match(question) for pattern in template_list)

    def make_prediction(self):
        question = self.question_entry.text().lower().strip()
        if not question:
            QtWidgets.QMessageBox.warning(self, title="Внимание", text="Введите вопрос")
            return

        if not self.current_image:
            QtWidgets.QMessageBox.warning(self, title="Внимание", text="Выберите изображение")
            return

        processed_ques = preprocess_data.preprocess_ques_type(question)
        ques_type = make_preds.make_preds_ques_type(processed_ques)

        if self.matches_pattern(question, self.multitarget_templates):
            processed_image, processed_label = preprocess_data.preprocess_multitarget(self.image_path, processed_ques)
            pred_label = make_preds.make_preds_multitarget(processed_image)
            self.prediction_output.setText(text=f"Predicted: {pred_label}")
        elif ques_type == 'yes/no':
            processed_image, processed_question = preprocess_data.preprocess_binary(self.image_path, processed_ques)
            pred_label = make_preds.make_preds_binary(processed_image, processed_question)
            self.prediction_output.setText(text=f"Predicted: {pred_label}")
        else:
            inputs_processed = preprocess_data.preprocess_generation(self.image_path, question)
            outputs = make_preds.make_generation(inputs_processed)
            answer = preprocess_data.decode_outputs(outputs)
            self.prediction_output.setText(text=f"Predicted: {answer}")

def main():
    app = QtWidgets.QApplication(sys.argv)
    main_window = VQAApp()
    main_window.show()
    app.exec()


if __name__ == "__main__":
    main()
