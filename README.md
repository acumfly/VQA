# Pathology Visual Question Answering system

## О проекте

Проект посвящен разработке приложения, позволяющего задавать вопросы медицинского характера к содержанию патологических снимков. Язык реализации - английский.

**Цель проекта** – с помощью методов обработки естественного языка и компьютерного зрения создать систему, способную анализировать вопрос к снимку с патологией и генерировать ответ на естественном языке. Данная задача в области машинного обучения называется также задачей Visual Question Answering (VQA) и относится к классу мультимодальных.

Источник данных - [платформа Hugging Face](https://huggingface.co/datasets/flaviagiammarino/path-vqa).

Блок-схема реализации выглядит следующим образом:
![image](https://github.com/user-attachments/assets/c4f609f7-8038-4852-a323-29579148a719)

Из-за ограниченности вычислительных ресурсов и возможностей :blush: на данный момент реализовано и обучено две из трех запланированных моделей (многоклассовой и бинарной классификации).

## Демонстрация приложения

Пользователи могут загружать изображения через диалог выбора файла и задавать вопросы к ним. Приложение обрабатывает тип вопроса и определяет, какую из моделей необходимо запустить. Для каждого типа вопроса предусмотрена отдельная логика обработки и предсказания.

Графический интерфейс разработан с помощью бибилиотеки Tkinter.

<div align="center">
  <video width="630" height="300" src="https://github.com/user-attachments/assets/a7b3d946-d817-429f-bf04-fbeeb8f6dcd8" autoplay muted controls></video>
</div>

## Описание моделей

### Модель бинарной классификации

Архитектура сети содержит две ветви: одна предназначена для обработки изображений, другая — для обработки текста. В конце обе ветви объединяются, и результаты проходят через несколько полносвязных слоев перед финальным предсказанием. 

![image](https://github.com/user-attachments/assets/e288176e-d503-4c9b-8a46-e9c1d903ace1)

*Accuracy* на тестовой выборке = 0.86


### Модель многоклассовой классификации

Модель многоклассовой классификации построена на базе предобученной ResNet50 без верхних слоев и адаптирована под классификацию на 22 класса.

*Accuracy* на тестовой выборке = 0.7
