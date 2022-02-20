from tensorflow.keras import utils 
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model

import numpy as np
import matplotlib.pyplot as plt


if __name__ == '__main__':

    #загружаем модель    
    model = load_model('mnist_weights.h5')

    #Берем картинку с Paint
    #загружаем в формате массива 28х28
    image = utils.load_img('Test.png', target_size=(28, 28), color_mode = 'grayscale')

    plt.imshow(image.convert('RGBA'))
    plt.show()

    real = 3 #введите число с картинки

    #конвертируем и нормализуем данные
    test = np.array(image)
    test = test.reshape(1, 784)
    test = 1 - test/255

    predict = model.predict(test)
    predict = np.argmax(predict)
    print("Predict:", predict)
    print('Real:', real)
