import tensorflow as tf
import sys
import os
import keras.layers
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
# Carregar os dados do conjunto "cats_vs_dogs" do TensorFlow
data_path = tf.keras.utils.get_file(
    'cats_vs_dogs.zip',
    'https://storage.googleapis.com/mledu-datasets/cats_and_dogs_filtered.zip',
    extract=True,
    cache_dir='./'
)

print(data_path)
data_dir = os.path.join(os.path.dirname(data_path), 'cats_and_dogs_filtered')

train_dir = os.path.join(data_dir, 'train')
validation_dir = os.path.join(data_dir, 'validation')
# Iniciar alguns atributos e parametros
# with_data_augmentation = True
# model_name = "cnn2"  # mlp, cnn, cnn2, cnn2_dp(with batch normalization)

data_aug_train = [True, False]
model_names = ["mlp", "cnn", "cnn2", "cnn2_dp"]

batch_size = 20
img_size = 128
epochs = 50

for with_data_augmentation in data_aug_train:
    for model_name in model_names:
        print(model_name)
        # Iniciar os iteradores de imagem
        train_image_data_generator = ImageDataGenerator(rescale=1. / 255)
        train_image_data_generator_with_augmentation = ImageDataGenerator(
            rescale=1. / 255,
            rotation_range=40,
            width_shift_range=0.2,
            height_shift_range=0.2,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True,
            fill_mode='nearest',
        )

        if with_data_augmentation:
            train_generator = train_image_data_generator_with_augmentation
        else:
            train_generator = train_image_data_generator
        train_iterator = train_generator.flow_from_directory(
            train_dir,
            target_size=(img_size, img_size),
            batch_size=batch_size,
            shuffle=True,
            class_mode='binary')

        validation_image_data_generator = ImageDataGenerator(rescale=1. / 255.0)
        validation_iterator = validation_image_data_generator.flow_from_directory(
            validation_dir,
            target_size=(img_size, img_size),
            batch_size=batch_size,
            shuffle=True,
            class_mode='binary')

        # %%
        # Iniciar, construir e treinar o modelo
        early_stop = EarlyStopping(
            patience=10,
            verbose=1,
        )
        learning_rate_reduction = ReduceLROnPlateau(
            monitor='val_accuracy',
            patience=2,
            verbose=1,
            factor=0.5,
            min_lr=0.00001
        )
        callbacks = [early_stop, learning_rate_reduction]

        # CNN2_DPOT
        model_cnn2_dpout = keras.Sequential([
            keras.layers.Conv2D(16, (3, 3), activation='relu', input_shape=(img_size, img_size, 3)),
            keras.layers.BatchNormalization(),
            keras.layers.MaxPool2D((2, 2)),
            keras.layers.Dropout(0.25),

            keras.layers.Conv2D(32, (3, 3), activation='relu'),
            keras.layers.BatchNormalization(),
            keras.layers.MaxPool2D((2, 2)),
            keras.layers.Dropout(0.25),

            keras.layers.Conv2D(64, (3, 3), activation='relu'),
            keras.layers.BatchNormalization(),
            keras.layers.MaxPool2D((2, 2)),
            keras.layers.Dropout(0.25),

            keras.layers.Flatten(),
            keras.layers.Dense(512, activation='relu'),
            keras.layers.BatchNormalization(),
            keras.layers.Dropout(0.5),
            keras.layers.Dense(1, activation='sigmoid')
        ])

        # CNN2
        model_cnn2 = keras.Sequential([
            keras.layers.Conv2D(16, (3, 3), activation='relu', input_shape=(img_size, img_size, 3)),
            keras.layers.BatchNormalization(),
            keras.layers.MaxPool2D((2, 2)),

            keras.layers.Conv2D(32, (3, 3), activation='relu'),
            keras.layers.BatchNormalization(),
            keras.layers.MaxPool2D((2, 2)),

            keras.layers.Conv2D(64, (3, 3), activation='relu'),
            keras.layers.BatchNormalization(),
            keras.layers.MaxPool2D((2, 2)),

            keras.layers.Flatten(),
            keras.layers.Dense(512, activation='relu'),
            keras.layers.BatchNormalization(),
            keras.layers.Dense(1, activation='sigmoid')
        ])

        # CNN
        model_cnn = keras.Sequential([
            keras.layers.Conv2D(16, (3, 3), activation='relu', input_shape=(img_size, img_size, 3)),
            keras.layers.MaxPool2D((2, 2)),

            keras.layers.Conv2D(32, (3, 3), activation='relu'),
            keras.layers.MaxPool2D((2, 2)),

            keras.layers.Conv2D(64, (3, 3), activation='relu'),
            keras.layers.MaxPool2D((2, 2)),

            keras.layers.Flatten(),
            keras.layers.Dense(512, activation='relu'),
            keras.layers.Dense(1, activation='sigmoid')
        ])

        # MLP
        model_mlp = keras.Sequential([
            keras.layers.Flatten(input_shape=(img_size, img_size, 3)),
            keras.layers.Dense(512, activation='relu'),
            keras.layers.Dense(1, activation='sigmoid')
        ])

        if model_name == "mlp":
            model = model_mlp
        elif model_name == "cnn":
            model = model_cnn
        elif model_name == "cnn2":
            model = model_cnn
        elif model_name == "cnn2_dp":
            model = model_cnn2_dpout
        else:
            print("unknown model name")
            sys.exit()
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        model.summary()

        history = model.fit(train_iterator,
                            validation_data=validation_iterator,
                            steps_per_epoch=train_iterator.n // batch_size,
                            epochs=epochs,
                            validation_steps=validation_iterator.n // batch_size,
                            callbacks=callbacks
                            )
        # %%
        # Avaliar o modelo
        test_loss, test_acc = model.evaluate(validation_iterator, steps=validation_iterator.n // batch_size)
        print('Test Loss:', test_loss)
        print('Test Accuracy:', test_acc)
        # %%
        predictions = (model.predict(validation_iterator) >= 0.5).astype(int)
        # %%
        cm = confusion_matrix(validation_iterator.labels, predictions, labels=[0, 1])
        clr = classification_report(validation_iterator.labels, predictions, labels=[0, 1], target_names=["CAT", "DOG"])

        results_folder = './results/'
        file_name_base = model_name

        plt.figure(figsize=(6, 6))
        sns.heatmap(cm, annot=True, fmt='g', vmin=0, cmap='Blues', cbar=False)
        plt.xticks(ticks=[0.5, 1.5], labels=["CAT", "DOG"])
        plt.yticks(ticks=[0.5, 1.5], labels=["CAT", "DOG"])
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.title("Confusion Matrix")
        metric = '-cfmatrix'
        name_file = file_name_base + metric
        if with_data_augmentation:
            name_file += '-dataaug'
        name_file = name_file + '.png'
        matrix_file_name = name_file
        plt.savefig(results_folder + matrix_file_name)
        plt.show()

        print("Classification Report:\n----------------------\n", clr)
        # %%
        acc = history.history['accuracy']
        val_acc = history.history['val_accuracy']
        y_len = range(len(acc))
        plt.plot(y_len, acc, 'b', label='Training accuracy')
        plt.plot(y_len, val_acc, 'r', label='Validation accuracy')
        bottom, top = plt.gca().get_ylim()
        if top > 3:
            plt.gca().set_ylim(bottom, 3)
        plt.title('Training and validation accuracy')
        plt.legend()
        metric = '-acc'
        file_name = file_name_base + metric
        if with_data_augmentation:
            file_name += '-dataaug'
        file_name += '.png'
        accuracy_img_name = file_name
        plt.savefig(results_folder + accuracy_img_name)
        plt.figure()

        loss = history.history['loss']
        val_loss = history.history['val_loss']
        plt.plot(y_len, loss, 'b', label='Training Loss')
        plt.plot(y_len, val_loss, 'r', label='Validation Loss')
        bottom, top = plt.gca().get_ylim()
        if top > 3:
            plt.gca().set_ylim(bottom, 3)
        plt.title('Training and validation loss')
        plt.legend()
        metric = '-loss'
        file_name = file_name_base + metric
        if with_data_augmentation:
            file_name += '-dataaug'
        file_name += '.png'
        loss_img_name = file_name
        plt.savefig(results_folder + loss_img_name)
        plt.show()

        pd.DataFrame(history.history).plot(figsize=(8, 5))
        plt.grid(True)
        bottom, top = plt.gca().get_ylim()
        if top > 3:
            plt.gca().set_ylim(bottom, 3)
        metric = '-acc_vs_loss'
        file_name = file_name_base + metric
        if with_data_augmentation:
            file_name += '-dataaug'
        acc_vs_loss_file_name = file_name
        acc_vs_loss_file_name += '.png'
        plt.savefig(results_folder + acc_vs_loss_file_name)
        plt.show()
        # %%
        # Append-adds at last
        model_def = model_name.upper()
        if with_data_augmentation:
            model_def += "-DATAAUG"
        resultsFile = open(results_folder + "RESULTS-" + model_def + ".md", "a")  # append mode
        resultsFile.write("## " + model_def + "\n\n")
        resultsFile.write("```\n")
        resultsFile.write("Test Accuracy: " + str(test_acc) + "\n")
        resultsFile.write("Test Loss: " + str(test_loss) + "\n")
        resultsFile.write("```\n\n")
        resultsFile.write("![](" + matrix_file_name + ")\n\n")
        resultsFile.write("```\n" + clr + "```\n\n")
        resultsFile.write("![](" + accuracy_img_name + ")\n\n")
        resultsFile.write("![](" + loss_img_name + ")\n\n")
        resultsFile.write("![](" + acc_vs_loss_file_name + ")\n\n")
        resultsFile.close()
