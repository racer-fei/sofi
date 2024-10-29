import numpy as np
import tensorflow as tf
from keras.models import load_model, save_model
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt
import os
import cv2

IMG_WIDTH = 256
IMG_HEIGHT = 256
IMG_CHANNELS = 3
TRAIN_PATH = "/Users/sofialinheira/Desktop/IC/tusimple_preprocessed/training/frames"
TRAIN_MASKS_PATH = '/Users/sofialinheira/Desktop/IC/tusimple_preprocessed/training/lane-masks'

train_image_files = os.listdir(TRAIN_PATH)
train_mask_files = os.listdir(TRAIN_MASKS_PATH)

images = []
masks = []

def apply_edge_detection(image):
    if image is None:
        raise ValueError("A imagem fornecida é None. Verifique o caminho da imagem.")

    if image.dtype != np.uint8:
        image = (image * 255).astype(np.uint8)  # Normaliza a imagem para 8 bits se necessário
    edges = cv2.Canny(image, 100, 200)
    return cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)  # Converte de volta para 3 canais

def apply_sharpening(image):
    kernel = np.array([[0, -1, 0],
                       [-1, 5, -1],
                       [0, -1, 0]])
    return cv2.filter2D(image, -1, kernel)

def apply_retinex(image):
    log_image = np.log1p(image)
    return cv2.normalize(log_image, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

# Função para realizar diversificação de dados
def augment_image(image, mask):
    angle = np.random.uniform(-15, 15)
    M = cv2.getRotationMatrix2D((IMG_WIDTH / 2, IMG_HEIGHT / 2), angle, 1)
    image = cv2.warpAffine(image, M, (IMG_WIDTH, IMG_HEIGHT))
    mask = cv2.warpAffine(mask, M, (IMG_WIDTH, IMG_HEIGHT))
    return image, mask

# Carregar e processar as imagens
# Carregar e processar as imagens e máscaras ao mesmo tempo para garantir correspondência
for image_file in train_image_files:
    if os.path.splitext(image_file)[1].lower() == ".jpg":
        image_path = os.path.join(TRAIN_PATH, image_file)
        mask_file = image_file  # Assume que a máscara correspondente possui o mesmo nome da imagem
        mask_path = os.path.join(TRAIN_MASKS_PATH, mask_file)

        if not os.path.exists(mask_path):  # Verifica se a máscara correspondente existe
            print(f"Mascara não encontrada para {image_file}. Pulando essa imagem.")
            continue

        # Processamento da imagem
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (IMG_WIDTH, IMG_HEIGHT))
        image = image / 255.0  # Normalização para [0, 1]

        # Processamento da máscara
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        mask = cv2.resize(mask, (IMG_WIDTH, IMG_HEIGHT))
        mask = mask / 255.0
        mask = np.expand_dims(mask, axis=-1)

        # Aplicar os processamentos e aumentar os dados
        edge_image = apply_edge_detection((image * 255).astype(np.uint8))
        sharpened_image = apply_sharpening(image)
        retinex_image = apply_retinex(image)

        # Adicionar as imagens e máscaras processadas
        images.extend([image, edge_image / 255.0, sharpened_image / 255.0, retinex_image / 255.0])
        masks.extend([mask, mask, mask, mask])  # Adiciona a mesma máscara correspondente para cada variação

# Converte listas para arrays
X_train = np.array(images)
Y_train = np.array(masks)

# Verifica novamente se os tamanhos coincidem
if X_train.shape[0] != Y_train.shape[0]:
    raise ValueError(f"Número de imagens: {X_train.shape[0]}, Número de máscaras: {Y_train.shape[0]}")

# Função de perda
def dice_loss(y_true, y_pred):
    y_true_f = tf.keras.backend.flatten(y_true)
    y_pred_f = tf.keras.backend.flatten(y_pred)
    intersection = tf.keras.backend.sum(y_true_f * y_pred_f)
    return 1 - (2. * intersection + 1) / (tf.keras.backend.sum(y_true_f) + tf.keras.backend.sum(y_pred_f) + 1)

# Criação da U-Net com camadas de atenção e profundidade aumentada
def create_unet_model(input_shape=(IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS)):
    inputs = tf.keras.layers.Input(input_shape)
    s = inputs

    # Contratação (encoder)
    c1 = tf.keras.layers.Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(s)
    c1 = tf.keras.layers.Dropout(0.1)(c1)
    c1 = tf.keras.layers.Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c1)
    p1 = tf.keras.layers.MaxPooling2D((2, 2))(c1)

    c2 = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p1)
    c2 = tf.keras.layers.Dropout(0.2)(c2)
    c2 = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c2)
    p2 = tf.keras.layers.MaxPooling2D((2, 2))(c2)

    c3 = tf.keras.layers.Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p2)
    c3 = tf.keras.layers.Dropout(0.3)(c3)
    c3 = tf.keras.layers.Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c3)
    p3 = tf.keras.layers.MaxPooling2D((2, 2))(c3)

    c4 = tf.keras.layers.Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p3)
    c4 = tf.keras.layers.Dropout(0.4)(c4)
    c4 = tf.keras.layers.Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c4)

    # Expansão (decoder)
    u5 = tf.keras.layers.Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(c4)
    u5 = tf.keras.layers.concatenate([u5, c3]) 
    c5 = tf.keras.layers.Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u5)
    c5 = tf.keras.layers.Dropout(0.3)(c5)
    c5 = tf.keras.layers.Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c5)

    u6 = tf.keras.layers.Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(c5)
    u6 = tf.keras.layers.concatenate([u6, c2])
    c6 = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u6)
    c6 = tf.keras.layers.Dropout(0.2)(c6)
    c6 = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c6)

    u7 = tf.keras.layers.Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(c6)
    u7 = tf.keras.layers.concatenate([u7, c1])
    c7 = tf.keras.layers.Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u7)
    c7 = tf.keras.layers.Dropout(0.1)(c7)
    c7 = tf.keras.layers.Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c7)

    outputs = tf.keras.layers.Conv2D(1, (1, 1), activation='sigmoid')(c7) 

    return tf.keras.models.Model(inputs=[inputs], outputs=[outputs])

unet_model = create_unet_model()
unet_model.compile(optimizer=tf.keras.optimizers.legacy.Adam(learning_rate=0.001), 
                   loss=[dice_loss], metrics=['accuracy'])

lr_scheduler = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-5, verbose=1)

epocas=10
lote=10
history = unet_model.fit(X_train, Y_train, validation_split=0.1, epochs=epocas, batch_size=lote, callbacks=[lr_scheduler])

def plot_learning_curves(history):
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Perda de Treinamento')
    plt.plot(history.history['val_loss'], label='Perda de Validação')
    plt.title('Curva de Perda')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(history.history['accuracy'], label='Acurácia de Treinamento')
    plt.plot(history.history['val_accuracy'], label='Acurácia de Validação')
    plt.title('Curva de Acurácia')
    plt.legend()
    plt.tight_layout()
    plt.show()

unet_model.save('/Users/sofialinheira/Desktop/IC/codigos_teste/network_results/unet_network.h5')
plot_learning_curves(history)
