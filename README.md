# SVHN Generation: VAE vs Diffusion

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0-red.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Генерация уличных номеров (SVHN) с помощью VAE и условной диффузионной модели (DDPM). SVHN представляет собой набор цветных фотографий номеров домов с Google Street View. Датасет является сложным для генерации из-за вариативности фона, освещения и углов съёмки. Обе модели условные, то есть генерируют картинки по заданной цифре (от 0 до 9). Диффузионная модель (DDPM) восстанавливает изображение из случайного шума, постепенно убирая его за 1000 шагов. VAE сжимает картинку в латентное пространство (10 параметров) и учится восстанавливать обратно. Для оценки качества сделан тест, в котором нужно отличить картинки сгенерированные диффузионной моделью от настоящих.

## Тест "Угадай ИИ"

👉 [Отличить ИИ от реальных номеров](https://david-z-ai.github.io/svhn-digits-generation/)

## Реальные (оригиналы SVHN)
![](preview_images/real/image_0046_class_1.png) ![](preview_images/real/image_0164_class_4.png) ![](preview_images/real/image_0360_class_9.png)

## VAE (реконструкция)
![](screenshots/vae1.png) ![](screenshots/vae2.png) ![](screenshots/vae3.png)

## Диффузия (генерация)
![](preview_images/diffusion/image_0040_class_1.png) ![](preview_images/diffusion/image_0164_class_4.png) ![](preview_images/diffusion/image_0397_class_9.png)

## Быстрый старт

```bash
# Клонируй репозиторий
git clone https://github.com/[ТВОЙ_НИК]/[НАЗВАНИЕ_РЕПО].git
cd [НАЗВАНИЕ_РЕПО]

# Установи зависимости
pip install -r requirements.txt

# Скачай веса модели (ссылка в разделе "Скачать весы") в папку weights/
# Или обучи сам:
python train_vae.py
python train_diffusion.py

# Сгенерируй примеры (появятся в samples/)
python generate.py --model diffusion --num 100 --class 5

# Запусти тест локально
cd test_ai && python -m http.server 8000
# Открой в браузере http://localhost:8000
