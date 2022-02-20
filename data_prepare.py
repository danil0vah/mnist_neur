from tensorflow.keras.datasets import mnist
from tensorflow.keras import utils 

import numpy as np
import pickle as pkl

if __name__ == '__main__':
    
    #подгружаем данные библиотеки MNIST
    (x_train_base, y_train_base), (x_test_base, y_test_base) = mnist.load_data() 

    #данные представлены в виде матриц 28x28
    # преобразуем их в вектора
    x_train = x_train_base.reshape(60000, 784)
    x_test = x_test_base.reshape(10000, 784)

    #нормализуем их в диапазон от 0 до 1
    x_train = x_train.astype('float32')
    x_train /= 255
    x_test = x_test.astype('float32')
    x_test /= 255

    #нормализуем выходные данные в вектор "вероятностей".
    y_train = utils.to_categorical(y_train_base, 10)
    y_test = utils.to_categorical(y_test_base, 10)

    #сохраним для последующей работы
    pkl.dump(x_train, open('x_train.pkl', 'wb'))
    pkl.dump(y_train, open('y_train.pkl', 'wb'))

    pkl.dump(x_test, open('x_test.pkl', 'wb'))
    pkl.dump(y_test, open('y_test.pkl', 'wb'))
