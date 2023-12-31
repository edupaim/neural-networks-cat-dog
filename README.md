# Machine learning to differentiate dog and cat

![](cat-vs-dogs-image.jpg)

## Explicação (pt-br)

A motivação para a escolha do problema em minha pesquisa foi buscar uma estratégia para facilitar meu aprendizado sobre
Machine Learning. Meu objetivo era avançar para problemas mais complexos, como a identificação do uso de EPI 
(Equipamento de Proteção Individual). Para alcançar esse objetivo, decidi experimentar o aprendizado de máquina com um
problema semelhante, mas de complexidade menor. Assim, o objetivo deste trabalho é construir um modelo de rede neural
que possa distinguir entre cachorros e gatos.

O dataset escolhido para a aprendizagem foi
o [Cats_Vs_Dogs](https://storage.googleapis.com/mledu-datasets/cats_and_dogs_filtered.zip). Esse dataset é dividido em
uma coleção balanceada de 2000 imagens para treino e 1000 imagens para validação.

Para o pré-processamento dos dados, foi utilizada a biblioteca Keras para carregar e redimensionar as imagens. Para
determinar o melhor parâmetro para o redimensionamento das imagens foi feita uma análise estatística básica no conjunto
de dados de imagens. No pré-processamento também é aplicada a técnica de normalização dos pixels, e o embaralhamento dos
dados para o treinamento.

No experimento, também é aplicada a técnica de aumento de dados (data augmentation) para melhorar a capacidade do modelo
de generalizar padrões. Para aplicar a técnica de aumento de dados, são passados alguns parâmetros para definir valores
de rotação, deslocamento horizontal e vertical, cisalhamento, zoom e espelhamento horizontal.

O projeto utiliza diferentes arquiteturas de redes neurais, como MLP e CNN, e variações dessas arquiteturas, como CNN2 (
no qual utiliza camadas de normalização em lote) e CNN2_DP (no qual aplica técnica de dropout).
Os modelos que aplicam variações arquiteturais de CNN possuem a camada de classificação similar ao modelo que aplica
arquitetura MLP. E as arquiteturas Convolucionais variam estendendo o modelo mais simples (CNN), aplicando novas
camadas. Logo será apresentado com detalhes a arquitetura mais completa (CNN2_DP).

* Camadas Convolucionais:
    * Primeira camada convolucional: Possui 16 filtros com tamanho de kernel 3x3, utilizando a função de ativação ReLU.
      Camadas convolucionais iniciais com um número menor de filtros (como 16) são responsáveis por capturar
      características mais simples, como bordas ou texturas básicas. À medida que avançamos nas camadas convolucionais,
      o número de filtros aumenta, permitindo a extração de características mais complexas e abstratas.
    * Camadas de Batch Normalization: Aplicadas após cada camada convolucional, visam normalizar as ativações dos
      neurônios, acelerando o treinamento e tornando a rede mais robusta.
    * Camadas de Max Pooling: Aplicadas após cada camada convolucional, têm o objetivo de reduzir a dimensionalidade das
      características extraídas, mantendo as informações mais relevantes. Utiliza uma janela de pooling com tamanho 2x2.
* Camadas de Dropout: Aplicadas após cada camada de Max Pooling, com um valor de 0.25. O Dropout desativa aleatoriamente
  um percentual de neurônios durante o treinamento, o que reduz o overfitting, aumenta a capacidade de generalização e
  melhora a robustez do modelo.
* Camada Flatten: A camada Flatten é responsável por transformar o tensor de saída das camadas convolucionais em um
  vetor unidimensional, preparando-o para ser alimentado nas camadas densas subsequentes.
* Camadas Densas:
    * Primeira camada densa: Possui 512 unidades (neurônios) com a função de ativação ReLU. Essa camada tem o objetivo
      de aprender representações mais complexas e abstratas das características extraídas pelas camadas convolucionais.
    * Camada de Batch Normalization: Aplicada após a primeira camada densa para normalizar as ativações dos neurônios.
    * Camada de Dropout: Aplicada após a camada de Batch Normalization com um valor de 0.5 para regularização.
* Camada de Saída:
    * Última camada densa: Possui uma única unidade de saída com a função de ativação sigmoidal. Essa camada produz um
      valor entre 0 e 1, representando a probabilidade de uma imagem pertencer à classe-alvo.

O treino é executado no projeto utilizando 50 épocas, e registrando funções como chamada de retorno para redução da
aprendizagem ao perceber um platô e parada precoce na falta da melhoria da métrica (val_loss).

O projeto realiza a validação dos resultados usando a matriz de confusão e o relatório de classificação (classification
report). As principais métricas analisadas no resultado são a acurácia e o f1-score. Pois, como o objetivo da
aprendizagem não é minimizar o erro de classificação específico para alguma classe, e sim a capacidade de acertar a
classe correta. Não foi aplicada a técnica de validação cruzada (cross-validation), entretanto é possível que essa
técnica avalie o desempenho de um modelo de aprendizado de máquina de forma mais robusta.

## Preparing data

For this experiment, the dataset "cats_and_dogs" from
`https://storage.googleapis.com/mledu-datasets/cats_and_dogs_filtered.zip` was used.
This dataset is stored as a cache at the root of the project directory.

This dataset is already divided into a training and validation set. The training set has 2000 images of dogs and cats,
divided in half. The validation set has 1000 images of dogs and cats, divided in half.

Is applied the re-scale on `ImageDataGenerator` with `rescale=1. / 255`. And is applied the re-size (256x256) and
shuffle
on image iterators.

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

### CNN2 (with batch normalization)

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

### CNN2_DP (with batch normalization and dropout)

```
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
```

## Training

Run with `50 epochs` and some callbacks to: reduce learning rate on plateau and stop early without increasing metric (
val_loss).

```
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

![](results/mlp-cfmatrix.png)

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
Test Accuracy: 0.7580000162124634
Test Loss: 0.8685814142227173
```

![](results/cnn2-cfmatrix.png)

```
              precision    recall  f1-score   support

         CAT       0.53      0.53      0.53       500
         DOG       0.53      0.52      0.52       500

    accuracy                           0.53      1000
   macro avg       0.53      0.53      0.53      1000
weighted avg       0.53      0.53      0.53      1000
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

### Results

#### MLP-DATAAUG

```
Test Accuracy: 0.6000000238418579
Test Loss: 0.6639021635055542
```

![](results/mlp-cfmatrix-dataaug.png)

```
              precision    recall  f1-score   support

         CAT       0.48      0.54      0.51       500
         DOG       0.48      0.42      0.45       500

    accuracy                           0.48      1000
   macro avg       0.48      0.48      0.48      1000
weighted avg       0.48      0.48      0.48      1000
```

#### CNN-DATAAUG

```
Test Accuracy: 0.7540000081062317
Test Loss: 0.4956442713737488
```

![](results/cnn-cfmatrix-dataaug.png)

```
              precision    recall  f1-score   support

         CAT       0.49      0.46      0.48       500
         DOG       0.49      0.52      0.51       500

    accuracy                           0.49      1000
   macro avg       0.49      0.49      0.49      1000
weighted avg       0.49      0.49      0.49      1000
```

#### CNN2-DATAAUG

```
Test Accuracy: 0.7829999923706055
Test Loss: 0.4312402009963989
```

![](results/cnn2-cfmatrix-dataaug.png)

```
              precision    recall  f1-score   support

         CAT       0.53      0.58      0.55       500
         DOG       0.54      0.49      0.51       500

    accuracy                           0.53      1000
   macro avg       0.53      0.53      0.53      1000
weighted avg       0.53      0.53      0.53      1000
```

#### CNN2_DP-DATAAUG

```
Test Accuracy: 0.6779999732971191
Test Loss: 0.62696772813797
```

![](results/cnn2_dp-cfmatrix-dataaug.png)

```
              precision    recall  f1-score   support

         CAT       0.50      0.69      0.58       500
         DOG       0.49      0.29      0.37       500

    accuracy                           0.49      1000
   macro avg       0.49      0.49      0.47      1000
weighted avg       0.49      0.49      0.47      1000
```
