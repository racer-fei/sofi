import cv2 as cv
import numpy as np
from keras.models import load_model

# Carregar o modelo U-Net
model = load_model('unet_network.h5')

def roi_bottom_half(frame):
    height, width = frame.shape[:2]
    return frame[height // 3:, :]

def process_frame(frame, model):
    frame = roi_bottom_half(frame)
    
    # Redimensiona o frame para o tamanho esperado pela U-Net
    frame_resized = cv.resize(frame, (256, 256))
    
    # Normaliza os valores dos pixels para [0, 1]
    frame_normalized = frame_resized / 255.0
    
    # Adiciona uma dimensão extra para que o modelo U-Net receba no formato correto
    input_frame = np.expand_dims(frame_normalized, axis=0)
    
    # Faz a predição usando a U-Net
    predicted_mask = model.predict(input_frame)
    
    # Remove a dimensão extra e redimensiona a máscara para o tamanho original
    predicted_mask = predicted_mask.squeeze()
    predicted_mask = cv.resize(predicted_mask, (frame.shape[1], frame.shape[0]))
    
    # Binariza a máscara (converte para 0 e 255)
    predicted_mask = (predicted_mask > 0.5).astype(np.uint8) * 255
    
    # Aplica a máscara à imagem original para destacar as faixas
    output = cv.bitwise_and(frame, frame, mask=predicted_mask)
    
    return output, predicted_mask

def main():
    video = cv.VideoCapture("/Users/sofialinheira/Desktop/IC/videos_teste/lane2.mp4")

    while True:
        ret, frame = video.read()

        if not ret:
            break

        output, mask = process_frame(frame, model)

        # Exibir o resultado
        cv.imshow("Predicted Mask - Lane Detection", output)

        # Sair do loop se a tecla 'q' for pressionada
        if cv.waitKey(1) & 0xFF == ord('q'):
            break

    # Liberar os recursos
    video.release()
    cv.destroyAllWindows()

if __name__ == "__main__":
    main()
