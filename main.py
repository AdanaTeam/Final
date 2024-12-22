import handlers, os
from keras.utils import to_categorical
from keras.layers import Input, Convolution2D, Flatten, Dense, Concatenate, MaxPool2D, BatchNormalization, Dropout
from keras.models import Model
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
import tensorflow as tf
from keras.models import load_model
from PIL import Image

ROOT_DATASET_PATH = 'CASIA2'

AU_FOLDER = os.path.join(ROOT_DATASET_PATH, 'Au')
TP_FOLDER = os.path.join(ROOT_DATASET_PATH, 'Tp')

X_ELA = []
X_HFN = []
Y = []
class_names = ['real', 'fake']

early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)


def main():
    while True:
        print("\nМеню:")
        print("1. Обучить нейросеть")
        print("2. Выбрать изображение для предсказания")
        choice = input("Введите номер действия (или 'exit' для выхода): ")
        
        if choice == '1':
            prepare_images()
            shuffle_data()
            model = get_model()
            history = train(model)
            show_history_plot(history)
            
        elif choice == '2':
            model = load_model('model.keras')
            image_path = input("Введите путь к изображению: ")
            display_image_with_prediction(image_path, model)
            
        elif choice.lower() == 'exit':
            break

def prepare_image_for_prediction(image_path):
    image_ela = handlers.ELA(image_path, quality=90, size=(128, 128))
    image_ela = image_ela.reshape(1, 128, 128, 3)

    image_hfn = handlers.HFN(image_path, quality=90, size=(128, 128))
    image_hfn = image_hfn.reshape(1, 128, 128, 1)

    return image_ela, image_hfn


def predict_single_image(image_path, model):
    image_ela, image_hfn = prepare_image_for_prediction(image_path)
    
    y_pred = model.predict([image_ela, image_hfn])
    y_pred_class = np.argmax(y_pred, axis=1)[0]
    confidence = np.amax(y_pred) * 100
    return class_names[y_pred_class], confidence

image_size = (128, 128)

def display_image_with_prediction(image_path, model):
    image = Image.open(image_path)
    plt.imshow(image)
    plt.axis('off')
    plt.show()
    image_class, confidence = predict_single_image(image_path, model)
    print(f'Prediction: Class: {image_class}, Confidence: {confidence:.2f}%')


def prepare_images():
    global X_ELA, X_HFN, Y, AU_FOLDER, TP_FOLDER, GROUNDTRUTH_FOLDER
    quality = 90
    size = (128, 128)
    
    print('Loading...')
    amount = 0
    for _, _, fileNames in os.walk(AU_FOLDER):
        for fileName in fileNames:
            amount += 1
    for _, _, fileNames in os.walk(TP_FOLDER):
        for fileName in fileNames:
            amount += 1
            
    count = 0
    for dirName, _, fileNames in os.walk(AU_FOLDER):
        for fileName in fileNames:
            count += 1
            fullPath = os.path.join(dirName, fileName)
            X_ELA.append(handlers.ELA(fullPath, quality, size))
            X_HFN.append(handlers.HFN(fullPath, quality, size))
            Y.append(0)
            if(count % 50 == 0):
                print("\033[A                             \033[A")
                print(f'{count}/{amount}')
                
    for dirName, _, fileNames in os.walk(TP_FOLDER):
        for fileName in fileNames:
            count += 1
            fullPath = os.path.join(dirName, fileName)

            X_ELA.append(handlers.ELA(fullPath, quality, size))
            X_HFN.append(handlers.HFN(fullPath, quality, size))
            Y.append(1)
            if(count % 50 == 0):
                print("\033[A                             \033[A")
                print(f'{count}/{amount}')
                
    X_ELA = np.array(X_ELA).reshape(-1, 128, 128, 3)
    X_HFN = np.array(X_HFN).reshape(-1, 128, 128, 1)
    Y = to_categorical(Y, 2)


def shuffle_data():
    global X_ELA, X_HFN, Y
    datas = [[X_ELA[i], X_HFN[i], Y[i]] for i in range(0, len(X_ELA))]
    datas = shuffle(datas, random_state=47)
    X_ELA = [data[0] for data in datas]
    X_HFN = [data[1] for data in datas]
    Y = [data[2] for data in datas]


def split_data():
    global X_ELA, X_HFN, Y
    x_ela_train, x_ela_test, y_train, y_test = train_test_split(X_ELA, Y, test_size = 0.2, train_size=0.8, random_state=47)
    x_hfn_train, x_hfn_test, _, _ = train_test_split(X_HFN, Y, test_size = 0.2, train_size=0.8, random_state=47)
    return np.array(x_ela_train), np.array(x_hfn_train), np.array(y_train), np.array(x_ela_test), np.array(x_hfn_test), np.array(y_test)


def get_cnn_model(input_shape):
    input_layer = Input(shape=input_shape)
    conv_layer = Convolution2D(32, (5, 5), activation='relu')(input_layer)
    pool_layer = MaxPool2D(pool_size = (2, 2))(conv_layer)
    drop_layer = Dropout(0.25)(pool_layer)
    flat_layer = Flatten()(drop_layer)
    dense_layer = Dense(128, activation='relu')(flat_layer)
    return [input_layer, dense_layer]


def get_model():
    cnn_model1 = get_cnn_model((128, 128, 3))
    cnn_model2 = get_cnn_model((128, 128, 1))
    conc_layer = Concatenate(axis=1)([cnn_model1[-1], cnn_model2[-1]])
    dense_layer = Dense(128, activation='relu')(conc_layer)
    drop_layer = Dropout(0.5)(dense_layer)
    output_layer = Dense(2, activation='softmax')(drop_layer)
    model = Model(inputs=[cnn_model1[0], cnn_model2[0]], outputs=output_layer)
    model.summary()
    return model

def train(model):
    x_ela_train, x_hfn_train, y_train, x_ela_test, x_hfn_test, y_test = split_data()
    epochs = 20
    batchSize = 64
    model.compile(
        optimizer = Adam(learning_rate=1e-4),
        loss = 'binary_crossentropy',
        metrics = ['accuracy'])
    hist = model.fit(
        [x_ela_train, x_hfn_train],
        y_train,
        batch_size = batchSize,
        epochs = epochs,
        callbacks=[early_stopping],
        validation_data = ([x_ela_test, x_hfn_test], y_test))
    model.save('model.keras')
    return hist


def show_history_plot(hist):
    plt.xlabel('Epoch')
    plt.ylabel('Metric')
    plt.plot(hist.history['loss'], color='maroon', label="Training loss")
    plt.plot(hist.history['val_loss'], color='red', label="Validation loss")
    plt.plot(hist.history['accuracy'], color='darkgreen', label="Training accuracy")
    plt.plot(hist.history['val_accuracy'], color='lime', label="Validation accuracy")
    plt.legend(loc='best', shadow=True)
    plt.show()


if __name__ == '__main__':
    main()