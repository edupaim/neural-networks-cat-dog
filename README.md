# Learn Machine to differentiate dog and cat

## Preparing data

For this experiment, the dataset "cats_and_dogs" from
`https://storage.googleapis.com/mledu-datasets/cats_and_dogs_filtered.zip` was used.
This dataset is stored as a cache at the root of the project directory.

This dataset is already divided into a training and validation set. The training set has 2000 images of dogs and cats,
divided in half. The validation set has 1000 images of dogs and cats, divided in half.

## Build Models

### MLP

```
model_mlp = keras.Sequential([
    keras.layers.Flatten(input_shape=(img_size, img_size, 3)),
    keras.layers.Dense(512, activation='relu'),
    keras.layers.Dense(1, activation='sigmoid')
])
```

### CNN

```
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
```

### CNN2

```
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
```

### Training history

| Model          |           Accuracy           |             Loss              |
|----------------|:----------------------------:|:-----------------------------:|
| MLP            |   ![](results/mlp-acc.png)   |   ![](results/mlp-loss.png)   |
| CNN            |   ![](results/cnn-acc.png)   |   ![](results/cnn-loss.png)   |
| CNN2           |  ![](results/cnn2-acc.png)   |  ![](results/cnn2-loss.png)   |
| CNN2 (DROPOUT) | ![](results/cnn2_dp-acc.png) | ![](results/cnn2_dp-loss.png) |

### Results

#### MLP
```
Test Accuracy: 0.5820000171661377
Test Loss: 0.74488365650177
```

![](mlp-cfmatrix.png)

```
              precision    recall  f1-score   support

         CAT       0.50      0.63      0.55       500
         DOG       0.49      0.36      0.41       500

    accuracy                           0.49      1000
   macro avg       0.49      0.49      0.48      1000
weighted avg       0.49      0.49      0.48      1000
```

#### CNN
```
Test Accuracy: 0.718999981880188
Test Loss: 1.1275080442428589
```

![](results/cnn-cfmatrix.png)

```
              precision    recall  f1-score   support

         CAT       0.49      0.50      0.49       500
         DOG       0.49      0.49      0.49       500

    accuracy                           0.49      1000
   macro avg       0.49      0.49      0.49      1000
weighted avg       0.49      0.49      0.49      1000
```

#### CNN2
```
Test Accuracy: 0.7120000123977661
Test Loss: 1.5042719841003418
```

![](results/cnn2-cfmatrix.png)

```
              precision    recall  f1-score   support

         CAT       0.52      0.53      0.53       500
         DOG       0.52      0.52      0.52       500

    accuracy                           0.52      1000
   macro avg       0.52      0.52      0.52      1000
weighted avg       0.52      0.52      0.52      1000
```

#### CNN2 with Dropout
```
Test Accuracy: 0.7490000128746033
Test Loss: 0.7401270866394043
```

![](results/cnn2_dp-cfmatrix.png)

```
              precision    recall  f1-score   support

         CAT       0.51      0.63      0.56       500
         DOG       0.51      0.39      0.44       500

    accuracy                           0.51      1000
   macro avg       0.51      0.51      0.50      1000
weighted avg       0.51      0.51      0.50      1000
```

## Data Augmentation

To generate the images iterator for model fit, function `ImageDataGenerator` from library
`keras.preprocessing.image` is used.

This function is called, passing some parameters to data augmentation as:

```
    rotation_range=40
    width_shift_range=0.2
    height_shift_range=0.2
    shear_range=0.2
    zoom_range=0.2
    horizontal_flip=True
    fill_mode='nearest'
```

### Training history with Data Augmentation

| Model          |               Accuracy               |                 Loss                  | 
|----------------|:------------------------------------:|:-------------------------------------:|
| MLP            |   ![](results/mlp-acc-dataaug.png)   |   ![](results/mlp-loss-dataaug.png)   |
| CNN            |   ![](results/cnn-acc-dataaug.png)   |   ![](results/cnn-loss-dataaug.png)   |
| CNN2           |  ![](results/cnn2-acc-dataaug.png)   |  ![](results/cnn2-loss-dataaug.png)   |
| CNN2 (DROPOUT) | ![](results/cnn2_dp-acc-dataaug.png) | ![](results/cnn2_dp-loss-dataaug.png) |