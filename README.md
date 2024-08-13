# Pathology Visual Question Answering system

## О проекте

Проект посвящен разработке приложения, позволяющего задавать вопросы медицинского характера к содержанию патологических снимков. Язык реализации - английский.

**Цель проекта** – с помощью методов обработки естественного языка и компьютерного зрения создать систему, способную анализировать вопрос к снимку с патологией и генерировать ответ на естественном языке. Данная задача в области машинного обучения называется также задачей Visual Question Answering (VQA) и относится к классу мультимодальных.

Источник данных - [платформа Hugging Face](https://huggingface.co/datasets/flaviagiammarino/path-vqa).

Блок-схема реализации выглядит следующим образом:
![image](https://github.com/user-attachments/assets/c4f609f7-8038-4852-a323-29579148a719)

Из-за ограниченности вычислительных ресурсов и возможностей :blush: на данный момент реализовано и обучено две из трех запланированных моделей (многоклассовой и бинарной классификации).

## Демонстрация приложения

Пользователи могут загружать изображения через диалог выбора файла и задавать вопросы о них. Приложение обрабатывает тип вопроса и определяет, какую из моделей необходимо запустить. Для каждого типа вопроса предусмотрена отдельная логика обработки и предсказания.

<div align="center">
  <video width="630" height="300" src="https://github.com/user-attachments/assets/a7b3d946-d817-429f-bf04-fbeeb8f6dcd8" controls></video>
</div>
