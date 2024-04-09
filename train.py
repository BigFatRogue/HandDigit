import os
import numpy as np
import matplotlib.pyplot as plt
from keras.datasets import mnist
import keras
from keras.optimizers import Adam
from keras.layers import Dense, Flatten


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Загрузка рукописных цифр. База mnist от Google
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Нормализация
x_train = x_train/255
x_test = x_test/255
#
# Преобразование выходных значений в векторы по категориям
# выход НС: [0, 0, 0, 0, 5, 0, 0, 0, 0, 0], что соответствует 5.
# в обучающий выборке значения представлены в виде цифр от 0 до 9
# и надо их преобразовать в вектор подобный выход НС
# 1 => [0, 1, 0, 0, 0, 0, 0, 0, 0]
y_train_cat = keras.utils.to_categorical(y_train, 10)
y_test_cat = keras.utils.to_categorical(y_test, 10)
#
# plt.figure(figsize=(10,5))
# for i in range(25):
#     plt.subplot(5, 5, i+1)
#     plt.xticks([])
#     plt.yticks([])
#     plt.imshow(x_train[i], cmap=plt.cm.binary)
#
# plt.show()


# Flatten - позволяет матрица 28х28 превратить в вектор 728
# Dense(128, activation='relu') - первый слой с функцией активации ReLu
# Dense(10, activation='softmax' - второй(последний) слой с ф-цией ак-вации Softmax
model = keras.Sequential([
    Flatten(input_shape=(28, 28, 1)),
    Dense(50, activation='relu'),
    Dense(10, activation='softmax')
])

#Вывод модели
# print(model.summary())

# Оптимизация по Adam, критерий качества категориальная кросс-энтропия
# metrics=['accuracy'] - метрика, которая нам нужна
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# validation_split - разбивает обучающую выборку на обучающую и выборку валидации
model.fit(x_train, y_train_cat, epochs=5, batch_size=30, validation_split=0.2, verbose=True)

# Проверка качества. Прогон тестовой выборки
model.evaluate(x_test, y_test_cat)
model.save('find_digit.h5')

# Проверка распознования
n = 0
# Добавляет новую ось, так как во время обучения мы передавали не одно изображение, а сразу пачку
x = np.expand_dims(x_test[0], axis=0)
res = model.predict(x)
print(res)
print(np.argmax(res))
#
# plt.imshow(x_test[n], cmap=plt.cm.binary)
# plt.show()
#
# # Распознование всей тестовой выборки
# pred = model.predict(x_test)
# # Индекс максимального значения
# pred = np.argmax(pred, axis=1)
#
# print(pred.shape)
#
# print(pred[:20])
# print(y_test[:20])
#
# # Выделение неверный вариантов
# # Сравниваем поэлементно каждый элемент с тестовой выборкой
# mask = pred == y_test
# print(mask[:10])
#
# # Оставим только те значения которые не False
# # ~ инверсия маски
# x_false = x_test[~mask]
# p_false = pred[~mask]
#
# # Кол-во неверно распознанных
# print(x_false.shape)

# for i in range(5):
#     print(str(p_false[i]))
#     plt.imshow(x_false[i], cmap=plt.cm.binary)
#     plt.show()
