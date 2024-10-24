import cv2 as cv
import numpy as np
from keras.models import load_model

# Carregar o modelo U-Net
model = load_model('unet_network.h5')

def roi_bottom_half(frame):
    height, width = frame.shape[:2]
    return frame[height // 3:, :]

def process_frame(frame, model, weight_model=0.7, weight_canny=0.3):
    # Seleciona a região de interesse
    frame = roi_bottom_half(frame)

    # Redimensiona o frame para o tamanho esperado pela U-Net e normaliza os valores dos pixels
    frame_resized = cv.resize(frame, (256, 256))
    frame_normalized = frame_resized / 255.0
    input_frame = np.expand_dims(frame_normalized, axis=0)

    # Predição da máscara usando o modelo U-Net
    predicted_mask = model.predict(input_frame)
    predicted_mask = np.squeeze(predicted_mask)
    predicted_mask = cv.resize(predicted_mask, (frame.shape[1], frame.shape[0]))

    # Binariza a máscara predita
    predicted_mask = (predicted_mask > 0.5).astype(np.uint8) * 255

    # Aplica a máscara predita à imagem original para destacar as faixas
    output = cv.bitwise_and(frame, frame, mask=predicted_mask)

    # Aplicar suavização e Canny para detectar bordas
    blurred = cv.GaussianBlur(output, (3, 3), 0)
    soft = 50
    bold = 15
    edges = cv.Canny(blurred, soft, bold)

    # Criar uma interseção ponderada entre a máscara predita e as bordas detectadas pelo Canny
    combined_mask = cv.addWeighted(predicted_mask.astype(np.float32), weight_model, edges.astype(np.float32), weight_canny, 0)
    combined_mask = (combined_mask > 127).astype(np.uint8) * 255  # Binariza novamente após combinação

    # Detecta as linhas usando a máscara combinada
    length = 50  # Ajustar para aumentar o mínimo comprimento da linha
    gap = 20     # Ajustar para aumentar o máximo intervalo entre as linhas
    lines = cv.HoughLinesP(combined_mask, 1, np.pi / 180, 50, minLineLength=length, maxLineGap=gap)

    # Desenhar as linhas detectadas no frame
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv.line(frame, (x1, y1), (x2, y2), (0, 255, 0), 5)

    return frame, combined_mask  # Retorna o frame com as linhas desenhadas e a máscara combinada

def main():
    video = cv.VideoCapture("/Users/sofialinheira/Desktop/IC/videos_teste/lane1.mp4")
    
    if not video.isOpened():  # Verifica se o vídeo foi aberto corretamente
        print("Erro ao abrir o vídeo.")
        return
    
    while True:
        ret, frame = video.read()
        if not ret:
            break

        # Processa o frame e combina as máscaras
        output, mask = process_frame(frame, model)
        cv.imshow("Linhas detectadas", output)
        cv.imshow("Mascara combinada", mask)

        if cv.waitKey(1) & 0xFF == ord('q'):
            break

    video.release()
    cv.destroyAllWindows()

if __name__ == "__main__":
    main()
