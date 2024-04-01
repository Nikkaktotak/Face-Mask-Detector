import sys
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
from sklearn.model_selection import train_test_split
from skimage import io, transform
from sklearn.preprocessing import label_binarize
import cv2
from keras.models import Sequential
from keras.layers import Dense, Flatten
from keras.optimizers import Adam


# Шлях до кореневої папки бази даних
root_dir = 'D:/pro/Face Mask Detector/FaceMaskDataset/data'

images = []
labels = []

# Функція для завантаження зображень та міток
def load_data(root_dir):
    for folder in os.listdir(root_dir):
        if os.path.isdir(os.path.join(root_dir, folder)):
            label = 1 if folder == 'with_mask' else 0
            for filename in os.listdir(os.path.join(root_dir, folder)):
                img_path = os.path.join(root_dir, folder, filename)
                # Зчитуємо зображення та змінюємо його розмір, якщо потрібно
                img = cv2.imread(img_path)
                img = cv2.resize(img, (224, 224))  # Розмір може бути іншим, в залежності від вашої моделі
                # Додаємо зображення та його мітку до списків
                images.append(img)
                labels.append(label)

    return np.array(images), np.array(labels)

# Завантаження даних
images, labels = load_data(root_dir)

# Розділення на тренувальний та тестувальний набори даних
X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.2, random_state=42)

# Конвертувати текстові мітки у відповідні вектори
y_train = label_binarize(y_train, classes=np.unique(y_train))
y_test = label_binarize(y_test, classes=np.unique(y_test))

# Модель нейронної мережі
model = Sequential([
    Flatten(input_shape=(224, 224, 3)),
    Dense(128, activation='relu'),
    Dense(64, activation='relu'),
    Dense(1, activation='sigmoid')
])

# Компіляція моделі
model.compile(optimizer=Adam(), loss='binary_crossentropy', metrics=['accuracy'])

# Навчання моделі
history = model.fit(X_train, y_train, epochs=15, validation_split=0.3)

# Оцінка точності на тестовому наборі
test_loss, test_accuracy = model.evaluate(X_test, y_test)
print(f"Test Accuracy: {test_accuracy}")

# Графіки навчання
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

# Зберігаємо графіки у файл
plt.savefig("graphics.png")
plt.show()