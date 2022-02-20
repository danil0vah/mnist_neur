from tensorflow.keras.models import load_model

import numpy as np
import pickle as pkl
import random


if __name__ == '___main__':

    #загружаем модель
    model = load_model('mnist_weights.h5')

    x_test = pkl.load(open('x_test.pkl','rb'))
    y_test = pkl.load(open('y_test.pkl','rb'))

    #проверяем точность модели на подготовленных тестовых данных 
    print(model.evaluate(x_test, y_test))

    #проверим наглядно на 10 радомно выбранных данных
    test_sample = list(zip(x_test, y_test))

    for (x, y) in random.sample(test_sample, 10):

        x = np.expand_dims(x, axis = 0)
        
        print("Predict:", np.argmax(model.predict(x)))
        print('Real:', np.argmax(y))