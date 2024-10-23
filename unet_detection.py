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
    input_frame = np.expand_dims(frame_normalized, axis=0)
    predicted_mask = model.predict(input_frame)
    
    # Remove a dimensão extra e redimensiona a máscara para o tamanho original
    predicted_mask = np.squeeze(predicted_mask)
    predicted_mask = cv.resize(predicted_mask, (frame.shape[1], frame.shape[0]))
    
    # Binariza a máscara (converte para 0 e 255)
    predicted_mask = (predicted_mask > 0.5).astype(np.uint8) * 255
    
    # Aplica a máscara à imagem original para destacar as faixas
    output = cv.bitwise_and(frame, frame, mask=predicted_mask)
    blurred = cv.GaussianBlur(output, (7, 7), 0)
    edges = cv.Canny(blurred, 75, 150)
    
    length = 5 
    gap = 9      
    lines = cv.HoughLinesP(edges, 1, np.pi / 180, 50, minLineLength=length, maxLineGap=gap)
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv.line(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)

    return frame, predicted_mask  # Retorna o frame com as linhas desenhadas
def main():
    video = cv.VideoCapture("/Users/sofialinheira/Desktop/IC/videos_teste/lane2.mp4")
    
    if not video.isOpened():  # Verifica se o vídeo foi aberto corretamente
        print("Erro ao abrir o vídeo.")
        return
    while True:
        ret, frame = video.read()
        if not ret:
            break

        output, mask = process_frame(frame, model)
        cv.imshow("Linhas detectadas", output)

        if cv.waitKey(1) & 0xFF == ord('q'):
            break

    video.release()
    cv.destroyAllWindows()

if __name__ == "__main__":
    main()
