import pandas as pd
import tkinter as tk
from tkinter import ttk, filedialog, Label, Entry, Button, Frame
from PIL import ImageTk, Image
import re
from preprocess_data import preprocess_multitarget, preprocess_binary, preprocess_generation, decode_outputs, preprocess_ques_type
from make_preds import make_preds_multitarget, make_preds_binary, make_generation, make_preds_ques_type


class VQAApp:
    def __init__(self, root):
        self.root = root
        self.image_path = None
        self.multitarget_test = pd.read_csv('datasets/multitarget_test.csv')
        self.binary_test = pd.read_csv('datasets/binary_test.csv')
        self.setup_ui()
        self.multitarget_ques = ['what is present?',]

    def center_window(self):
        screen_width = self.root.winfo_screenwidth()
        screen_height = self.root.winfo_screenheight()
        window_width = 500
        window_height = 500  # Увеличено для размещения всех элементов
        x_cordinate = int((screen_width / 2) - (window_width / 2))
        y_cordinate = int((screen_height / 2) - (window_height / 2))
        self.root.geometry(f"{window_width}x{window_height}+{x_cordinate}+{y_cordinate}")

    def setup_ui(self):
        self.root.update_idletasks()
        self.root.update()
        self.root.title('Image and Question')
        style = ttk.Style()
        style.theme_use('clam')

        self.image_frame = Frame(self.root)
        self.image_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        control_frame = Frame(self.root)
        control_frame.pack(side=tk.TOP, fill=tk.X) #, pady=10

        self.question_label = Label(control_frame, text='Enter your question:')
        self.question_label.pack(side=tk.LEFT, padx=5)

        self.question_entry = Entry(control_frame, width=50)
        self.question_entry.pack(side=tk.LEFT, padx=5)
        self.question_entry.focus_set()

        self.submit_button = Button(control_frame, text='Submit', command=self.on_submit)
        self.submit_button.pack(side=tk.LEFT, padx=5)

        self.result_label = Label(self.root)
        self.result_label.pack(pady=10)

        self.choose_another_file_button = Button(self.root, text='Choose another file', command=self.change_image)
        self.choose_another_file_button.pack(pady=5)

        self.center_window()
        self.image_path = self.open_image_file()

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
            self.result_label.config(text="Please enter a question.")
            return

        processed_ques = preprocess_ques_type(question)
        ques_type = make_preds_ques_type(processed_ques)
        print(ques_type)

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
        self.result_label.config(text="")
        self.question_entry.delete(0, tk.END)
        new_image_path = filedialog.askopenfilename(title="Select Image",
                                                    filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp")])
        if new_image_path:
            self.image_path = new_image_path
            self.update_image_display(new_image_path)

    def update_image_display(self, image_path):
        if hasattr(self, 'image_label') and self.image_label.winfo_exists():
            self.image_label.destroy()

        image = Image.open(image_path)
        max_size = (self.image_frame.winfo_width(), self.image_frame.winfo_height())
        image.thumbnail(max_size, Image.LANCZOS)
        self.photo = ImageTk.PhotoImage(image)
        self.image_label = Label(self.image_frame, image=self.photo)
        self.image_label.pack(expand=True)

        self.root.update()

    def open_image_file(self):
        image_path = filedialog.askopenfilename(title="Select Image",
                                                initialdir='D:\images\image_folder',
                                                filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp")])
        if not image_path:
            print("No image selected.")
            self.root.destroy()
            return
        self.image_path = image_path
        self.update_image_display(image_path)
        return image_path

    def run(self):
        self.root.mainloop()


if __name__ == "__main__":
    root = tk.Tk()
    app = VQAApp(root)
    app.run()
