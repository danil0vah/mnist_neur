from tensorflow.keras import utils 
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model

from PIL import Image

import numpy as np
import matplotlib.pyplot as plt


if __name__ == '__main__':

    #загружаем модель    
    model = load_model('mnist_weights.h5')

    #Берем фотографию
    image = utils.load_img('Test2.jpg', target_size=(28, 28), color_mode = 'grayscale')

    plt.imshow(image.convert('RGBA'))
    plt.show()

    real = 7 #введите число с картинки


    #преобразуем картинку в вектор-массив и преобразуем
    test = np.array(image)
    test = test.reshape(1, 784)
    test = np.where(test > 100, 255, test)

    #проверим
    plt.imshow(Image.fromarray(test.reshape(28,28)).convert('RGBA'))
    plt.show()

    #нормализуем и инвертируем
    test = 1 - test/255


    predict = model.predict(test)
    predict = np.argmax(predict)
    print("Predict:", predict)
    print('Real:', real)