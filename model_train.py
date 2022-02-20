from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam

import numpy as np
import pickle as pkl


if __name__ == '__main__':

    x_train = pkl.load(open('x_train.pkl','rb'))
    y_train = pkl.load(open('y_train.pkl','rb'))

    #создаем модель
    model = Sequential()
    model.add(Dense(800, input_dim = 784))
    model.add(Dense(400))
    model.add(Dense(10, activation = 'softmax'))

    model.summary()#общие данные модели

    model.compile(loss = 'categorical_crossentropy', optimizer=Adam(lr = 0.001), metrics=['accuracy'])
    # категориальная кроссэнтропия делает наши векторы предсказания как можно ближе к векторам выходных данных
    # оптимизатор Adam - оптимальный вариант
    #метрика - нужна нам для оценки обучения модели

    model.fit(x_train, y_train, batch_size = 1000, epochs = 15, verbose = 1, validation_split = 0.2)
    #подаем наши массивы входных и выходных данных
    #batch size - разбивает пакеты наших данных на партии
    #epochs - разделяет обучение на фазы и позволяет делать периодическую оценку
    #verbose - просто делает полный вывод данных процесса обучения
    #validation split - делит наши данные на тренировочные и проверочные в процессе обучения

    model.save_weights('mnist_weights.h5')#сохраним веса

