import pandas as pd
pd.options.mode.chained_assignment = None  # Suppress the warning
import os
import re
import hashlib

def prepare_datasets():
    df_train = pd.read_csv('D:/diploma/kaggle_data/df_train.csv')
    df_valid = pd.read_csv('D:/diploma/kaggle_data/df_valid.csv')
    df_test = pd.read_csv('D:/diploma/kaggle_data/df_test.csv')

    # Путь к папке images
    images_path = 'D:/images'

    # Создаем списки для хранения путей к файлам в каждой подпапке
    image_paths_train = []
    image_paths_test = []
    image_paths_valid = []

    # Обходим все поддиректории в папке images
    for root, dirs, files in os.walk(images_path):
        # Проверяем, является ли текущая поддиректория одной из требуемых
        if 'train' in root:
            # Добавляем пути к файлам в список train_files
            for file in files:
                image_paths_train.append(os.path.join(root, file))
        elif 'test' in root:
            # Добавляем пути к файлам в список test_files
            for file in files:
                image_paths_test.append(os.path.join(root, file))
        elif 'valid' in root:
            # Добавляем пути к файлам в список valid_files
            for file in files:
                image_paths_valid.append(os.path.join(root, file))

    image_paths_train = [path.replace('\\', '/') for path in image_paths_train]
    image_paths_test = [path.replace('\\', '/') for path in image_paths_test]
    image_paths_valid = [path.replace('\\', '/') for path in image_paths_valid]

    # Функция для извлечения последней цифры из имени файла
    def extract_last_digit(filename):
        match = re.search(r'\d+(?=\.jpg$)', filename)
        return int(match.group()) if match else 0

    # Сортировка списка файлов по последней цифре в их именах
    image_paths_train = sorted(image_paths_train, key=extract_last_digit)
    image_paths_test = sorted(image_paths_test, key=extract_last_digit)
    image_paths_valid = sorted(image_paths_valid, key=extract_last_digit)

    df_train['image_path'] = image_paths_train
    df_valid['image_path'] = image_paths_valid
    df_test['image_path'] = image_paths_test



    classification_train = df_train[((df_train['question'].str.contains(r'is .* present\?', regex=True)) \
                                     & (df_train['answer'] == 'yes'))]
    classification_valid = df_valid[((df_valid['question'].str.contains(r'is .* present\?', regex=True)) \
                                     & (df_valid['answer'] == 'yes'))]
    classification_test = df_test[(df_test['question'].str.contains(r'is .* present\?', regex=True)) \
                                  & (df_test['answer'] == 'yes')]

    def extract_class_label(row):
      # Используем регулярное выражение для поиска второго слова в вопросе
      class_match = re.search(r'is (\w+) present\?', row['question'])
      if class_match:
        return class_match.group(1)
      return None

    classification_train['target'] = classification_train.apply(extract_class_label, axis=1)
    classification_valid['target'] = classification_valid.apply(extract_class_label, axis=1)
    classification_test['target'] = classification_test.apply(extract_class_label, axis=1)

    classification_train_2 = df_train[df_train['question'] == 'what is present?']
    classification_valid_2 = df_valid[df_valid['question'] == 'what is present?']
    classification_test_2 = df_test[df_test['question'] == 'what is present?']

    classification_train_2['target'] = classification_train_2['answer']
    classification_valid_2['target'] = classification_valid_2['answer']
    classification_test_2['target'] = classification_test_2['answer']

    df1_train = classification_train.drop(['answer'], axis=1)
    df1_valid = classification_valid.drop(['answer'], axis=1)
    df1_test = classification_test.drop(['answer'], axis=1)

    df2_train = classification_train_2.drop(['answer'], axis=1)
    df2_valid = classification_valid_2.drop(['answer'], axis=1)
    df2_test = classification_test_2.drop(['answer'], axis=1)

    class_train = pd.concat([df1_train, df2_train])
    class_valid = pd.concat([df1_valid, df2_valid])
    class_test = pd.concat([df1_test, df2_test])

    classes = ['cardiovascular', 'thymus', 'vasculature', 'spleen', 'oral', 'joints', 'hepatobiliary', 'soft tissue',
     'hematologic', 'endocrine', 'liver', 'nervous', 'lymph node', 'blood', 'gastrointestinal', 'uterus', 'adrenal',
     'abdomen', 'bone marrow', 'female reproductive', 'pituitary', 'respiratory']

    multitarget_train = class_train[class_train['target'].isin(classes)]
    multitarget_valid = class_valid[class_valid['target'].isin(classes)]
    multitarget_test = class_test[class_test['target'].isin(classes)]

    multitarget_train.to_csv('multitarget_train.csv', index=False)
    multitarget_valid.to_csv('multitarget_valid.csv', index=False)
    multitarget_test.to_csv('multitarget_test.csv', index=False)

    df_train.to_csv('df_train.csv', index=False)
    df_valid.to_csv('df_valid.csv', index=False)
    df_test.to_csv('df_test.csv', index=False)

    bin_train = df_train[df_train['answer'].isin(['yes', 'no'])]
    bin_valid = df_valid[df_valid['answer'].isin(['yes', 'no'])]
    bin_test = df_test[df_test['answer'].isin(['yes', 'no'])]

    binary_train = bin_train
    binary_valid = bin_valid
    binary_test = bin_test

    binary_train.to_csv('binary_train.csv', index=False)
    binary_valid.to_csv('binary_valid.csv', index=False)
    binary_test.to_csv('binary_test.csv', index=False)

prepare_datasets()
