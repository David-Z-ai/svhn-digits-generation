# SVHN Generation: VAE vs Diffusion

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0-red.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Генерация уличных номеров (SVHN) с помощью VAE и условной диффузионной модели.  
В комплекте – интерактивный тест "Угадай ИИ": нужно отличить реальные картинки от сгенерированных.

## Результаты

| Модель       | FID     | Параметры | Время генерации 1000 img |
|--------------|---------|-----------|--------------------------|
| VAE          | 45.2    | 12M       | 2 секунды                |
| Diffusion    | 18.7    | 35M       | 2 минуты                 |

**Примеры:**

| Оригинал (SVHN) | Реконструкция VAE | Генерация диффузией |
|----------------|-------------------|---------------------|
| ![](screenshots/real.png) | ![](screenshots/vae_rec.png) | ![](screenshots/diff_gen.png) |

## Тест "Угадай ИИ"

Открой файл `test_ai/index.html` в браузере. Тебе покажут 10 картинок (половина реальных, половина сгенерированных). Угадай, какие нарисовала нейросеть.

![](screenshots/test_preview.png)

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
