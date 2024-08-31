import sys
import os
import PyQt5; dirname = os.path.dirname(__file__);
plugin_path = os.path.join(dirname, 'platforms');   os.environ['QT_QPA_PLATFORM_PLUGIN_PATH'] = plugin_path;
os.environ['QT_QPA_PLATFORM_PLUGIN_PATH'] = plugin_path
import pandas as pd
import re
from PyQt5.QtWidgets import QApplication, QLabel, QMainWindow, QPushButton, QVBoxLayout, QWidget, QFileDialog, QLineEdit, QFrame
from PyQt5.QtGui import QPixmap
from PyQt5.QtCore import Qt
from PIL import ImageTk, Image
from preprocess_data import preprocess_multitarget, preprocess_binary, preprocess_generation, decode_outputs, preprocess_ques_type
from make_preds import make_preds_multitarget, make_preds_binary, make_generation, make_preds_ques_type


class VQAApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.image_path = None
        self.multitarget_test = pd.read_csv('datasets/multitarget_test.csv')
        self.binary_test = pd.read_csv('datasets/binary_test.csv')
        self.init_ui()
        self.multitarget_ques = ['what is present?',]

    def init_ui(self):
        self.setWindowTitle('Image and Question')
        self.setGeometry(100, 100, 500, 600)

        self.central_widget = QWidget(self)
        self.setCentralWidget(self.central_widget)

        self.layout = QVBoxLayout(self.central_widget)

        self.image_frame = QFrame(self)
        self.layout.addWidget(self.image_frame)

        self.image_label = QLabel(self)
        self.image_label.setAlignment(Qt.AlignCenter)
        self.layout.addWidget(self.image_label)

        self.question_label = QLabel('Enter your question: ', self)
        self.layout.addWidget(self.question_label)

        self.question_entry = QLineEdit(self)
        self.layout.addWidget(self.question_entry)

        self.submit_button = QPushButton('Submit', self)
        self.submit_button.clicked.connect(self.on_submit)
        self.layout.addWidget(self.submit_button)

        self.result_label = QLabel("", self)
        self.result_label.setAlignment(Qt.AlignCenter)
        self.layout.addWidget(self.result_label)

        self.choose_another_file_button = QPushButton('Choose another file', self)
        self.choose_another_file_button.clicked.connect(self.change_image)
        self.layout.addWidget(self.choose_another_file_button)

        self.open_image_file()

    def get_info(self):
        multitarget_rows = self.multitarget_test[self.multitarget_test['image_path'] == self.image_path]
        binary_rows = self.binary_test[self.binary_test['image_path'] == self.image_path]
        info = dict()
        if multitarget_rows.empty and binary_rows.empty:
            return dict()
        if not multitarget_rows.empty:
            info['dataset'] = 'multitarget'
            image_hash = multitarget_rows.iloc[0]['image_hash']
            questions = multitarget_rows['question'].tolist()
            answers = multitarget_rows['target'].tolist()
        elif not binary_rows.empty:
            info['dataset'] = 'binary'
            image_hash = binary_rows.iloc[0]['image_hash']
            questions = binary_rows['question'].tolist()
            answers = binary_rows['answer'].tolist()
        info['image_hash'] = image_hash
        info['questions'] = questions
        info['answers'] = answers
        return info

    def prepro_question(self):
        question = self.question_entry.get().lower()
        question = question.replace('?', '')
        question = re.sub(r'[^\w\s]', '', question)
        question = re.sub(r'\s+', ' ', question).strip()
        return question

    def is_binary_question(self, question):
        regexp_bin1 = re.compile(r'is(.*?)(?=present)')
        regexp_bin2 = re.compile(r'does this image show .*')
        regexp_bin3 = re.compile(r'are(.*?)(?=present)')
        return regexp_bin1.match(question) or regexp_bin2.match(question) or regexp_bin3.match(question)

    def on_submit(self):
        multitarget_templates = ['what is present', 'what does this image show']

        question = self.prepro_question()

        if not question:  # Проверка на пустой ввод
            self.result_label.setText("Please enter a question.")
            return

        processed_ques = preprocess_ques_type(question)
        ques_type = make_preds_ques_type(processed_ques)

        if question in multitarget_templates:
            processed_image, processed_label = preprocess_multitarget(self.image_path, question)
            pred_label = make_preds_multitarget(processed_image)
            self.result_label.config(text=f"Predicted: {pred_label}")
        elif ques_type == 'yes/no':
            processed_image, processed_question = preprocess_binary(self.image_path, question)
            pred_label = make_preds_binary(processed_image, processed_question)
            self.result_label.config(text=f"Predicted: {pred_label}")
        else:
            inputs_processed = preprocess_generation(self.image_path, question)
            outputs = make_generation(inputs_processed)
            answer = decode_outputs(outputs)
            self.result_label.config(text=f"Predicted: {answer}")

    def change_image(self):
        self.result_label.setText("")
        self.question_entry.clear()
        new_image_path = QFileDialog.getOpenFileName(self,
                                                     "Select Image",
                                                     "",
                                                    "Image files (*.jpg *.jpeg *.png *.bmp)")
        if new_image_path:
            self.image_path = new_image_path
            self.update_image_display(new_image_path)

    def update_image_display(self, image_path):
        image = Image.open(image_path)
        image.thumbnail((self.image_label.width(), self.image_label.height()), Image.LANCZOS)
        image.save('temp_image.png')
        self.image_label.setPixmap(QPixmap('temp_image.ong'))

    def open_image_file(self):
        image_path, _ = QFileDialog.getOpenFileName(self,
                                                    "Select Image",
                                                    "D:\images\image_folder",
                                                    "Image files (*.jpg *.jpeg *.png *.bmp)")
        if not image_path:
            print("No image selected.")
            self.close()
            return
        self.image_path = image_path
        self.update_image_display(image_path)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    main_window = VQAApp()
    main_window.show()
    sys.exit(app.exec_())
