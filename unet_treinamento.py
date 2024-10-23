import numpy as np
import tensorflow as tf
from keras.models import load_model, save_model
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt
import os
import cv2

# Configurações
IMG_WIDTH = 256
IMG_HEIGHT = 256
IMG_CHANNELS = 3
TRAIN_PATH = "/Users/sofialinheira/Desktop/IC/tusimple_preprocessed/training/frames"
TRAIN_MASKS_PATH = '/Users/sofialinheira/Desktop/IC/tusimple_preprocessed/training/lane-masks'

# Carregamento das imagens e máscaras
train_image_files = os.listdir(TRAIN_PATH)
train_mask_files = os.listdir(TRAIN_MASKS_PATH)

X_train = np.zeros((len(train_image_files), IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS), dtype=np.float32)
Y_train = np.zeros((len(train_image_files), IMG_HEIGHT, IMG_WIDTH, 1), dtype=np.float32)

images = []
masks = []

for n, image_file in enumerate(train_image_files):
    if os.path.splitext(image_file)[1].lower() == ".jpg":
        image_path = os.path.join(TRAIN_PATH, image_file)
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (IMG_WIDTH, IMG_HEIGHT))
        image = image / 255.0  # Normalização para [0, 1]
        images.append(image)

for n, mask_file in enumerate(train_mask_files):
    if os.path.splitext(mask_file)[1].lower() == ".jpg":
        mask_path = os.path.join(TRAIN_MASKS_PATH, mask_file)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        mask = cv2.resize(mask, (IMG_WIDTH, IMG_HEIGHT))
        mask = mask / 255.0  # Normalização para [0, 1]
        masks.append(np.expand_dims(mask, axis=-1))

X_train, Y_train = np.array(images), np.array(masks)

# Criação da U-Net
def create_unet_model(input_shape=(IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS)):
    print("Criação da rede")
    inputs = tf.keras.layers.Input(input_shape)
    s = inputs

    # Camada de contração
    print("Camada de contração")
    c1 = tf.keras.layers.Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(s)
    c1 = tf.keras.layers.Dropout(0.1)(c1)
    c1 = tf.keras.layers.Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c1)
    p1 = tf.keras.layers.MaxPooling2D((2, 2))(c1)

    c2 = tf.keras.layers.Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p1)
    c2 = tf.keras.layers.Dropout(0.1)(c2)
    c2 = tf.keras.layers.Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c2)
    p2 = tf.keras.layers.MaxPooling2D((2, 2))(c2)

    c3 = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p2)
    c3 = tf.keras.layers.Dropout(0.2)(c3)
    c3 = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c3)
    p3 = tf.keras.layers.MaxPooling2D((2, 2))(c3)

    c4 = tf.keras.layers.Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p3)
    c4 = tf.keras.layers.Dropout(0.2)(c4)
    c4 = tf.keras.layers.Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c4)
    p4 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(c4)

    c5 = tf.keras.layers.Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p4)
    c5 = tf.keras.layers.Dropout(0.3)(c5)
    c5 = tf.keras.layers.Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c5)

    # Camada expansiva
    print("Camada expansiva")
    u6 = tf.keras.layers.Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(c5)
    u6 = tf.keras.layers.concatenate([u6, c4])
    c6 = tf.keras.layers.Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u6)
    c6 = tf.keras.layers.Dropout(0.2)(c6)
    c6 = tf.keras.layers.Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c6)

    u7 = tf.keras.layers.Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(c6)
    u7 = tf.keras.layers.concatenate([u7, c3])
    c7 = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u7)
    c7 = tf.keras.layers.Dropout(0.2)(c7)
    c7 = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c7)

    u8 = tf.keras.layers.Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(c7)
    u8 = tf.keras.layers.concatenate([u8, c2])
    c8 = tf.keras.layers.Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u8)
    c8 = tf.keras.layers.Dropout(0.1)(c8)
    c8 = tf.keras.layers.Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c8)

    u9 = tf.keras.layers.Conv2DTranspose(16, (2, 2), strides=(2, 2), padding='same')(c8)
    u9 = tf.keras.layers.concatenate([u9, c1], axis=3)
    c9 = tf.keras.layers.Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u9)
    c9 = tf.keras.layers.Dropout(0.1)(c9)
    c9 = tf.keras.layers.Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c9)

    outputs = tf.keras.layers.Conv2D(1, (1, 1), activation='sigmoid')(c9)

    model = tf.keras.Model(inputs=[inputs], outputs=[outputs])
    return model

# Criar e compilar o modelo U-Net
print("Compilando a rede")
unet_model = create_unet_model()
unet_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Treinar o modelo
history = unet_model.fit(X_train, Y_train, validation_split=0.1, epochs=2, batch_size=16)  # 50 épocas
print("Modelo de rede treinado")

# Salvar o modelo treinado
unet_model.save('/Users/sofialinheira/Desktop/IC/codigos_teste/neural_network/unet_network.h5')
unet_model.save('/Users/sofialinheira/Desktop/IC/codigos_teste/neural_network/unet.keras')

print("Modelo de rede salvo")

# -------- Avaliação da rede neural -----------
# Prever as máscaras de teste
previsoes = unet_model.predict(X_train)
previsoes_binarizadas = (previsoes > 0.5).astype(np.uint8)
Y_train_binarized = (Y_train > 0.5).astype(np.uint8)
y_train_flat = Y_train_binarized.flatten()
previsoes_flat = previsoes_binarizadas.flatten()

# Avaliar a acurácia
acc = accuracy_score(y_train_flat, previsoes_flat)
print(f"Accuracy: {acc}")

# Matriz de confusão
cm = confusion_matrix(y_train_flat, previsoes_flat)
print("Confusion Matrix")
print(cm)

# Relatório de classificação
print(classification_report(y_train_flat, previsoes_flat))

# Plotar a matriz de confusão
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.title("Matriz de Confusão")
plt.xlabel("Predito")
plt.ylabel("Real")
plt.savefig('/Users/sofialinheira/Desktop/IC/codigos_teste/neural_network/confusion_matrix.jpg')  # Salvar a matriz de confusão como imagem
plt.show()  # Exibir a matriz de confusão

# Exibir a acurácia como um gráfico simples
plt.figure(figsize=(6, 4))
plt.bar(['Accuracy'], [acc], color='green')
plt.ylim(0, 1)  # Limitar o eixo Y de 0 a 1
plt.title("Acurácia do Modelo")
plt.ylabel("Acurácia")
plt.savefig('/Users/sofialinheira/Desktop/IC/codigos_teste/neural_network/accuracy_score.jpg')  # Salvar a acurácia como imagem
plt.show()  # Exibir o gráfico da acurácia

# Criar um mapa de calor para visualização das previsões
plt.figure(figsize=(8, 6))
heat = sns.heatmap(previsoes_binarizadas.reshape(Y_train.shape[0], -1), cmap='Blues', cbar=True)
plt.title("Mapa de Calor das Previsões")
plt.xlabel("Pixels")
plt.ylabel("Imagens")
plt.savefig('/Users/sofialinheira/Desktop/IC/codigos_teste/neural_network/heat_map.jpg')  # Salvar o mapa de calor como imagem
plt.show()  # Exibir o mapa de calor
