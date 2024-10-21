import cv2 as cv
import numpy as np

def roi_bottom_half(frame):
    height, width = frame.shape[:2]
    return frame[height // 2:, :]
# Função para processar e exibir a cor branca
def process_frame(frame):
    frame = roi_bottom_half(frame)
    # Converter o frame de BGR para HSV
    hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)

    # Definir intervalo para a cor branca no espaço HSV
    lower_white = np.array([0, 0, 200])   # Baixo valor de saturação e alto valor
    upper_white = np.array([180, 55, 255]) # Saturação baixa e valor alto

    # Criar máscara para isolar a cor branca
    mask = cv.inRange(hsv, lower_white, upper_white)

    # Aplicar a máscara para extrair a cor branca da imagem original
    output = cv.bitwise_and(frame, frame, mask=mask)
    return output, mask

def main():
    # Abrir o vídeo
    video = cv.VideoCapture("/Users/sofialinheira/Desktop/IC/codigos_teste/videos/lane_video.mp4")

    while True:
        ret, frame = video.read()

        # Verificar se o frame foi capturado corretamente
        if not ret:
            break

        # Processar o frame para detectar a cor branca
        output, mask = process_frame(frame)

        # Exibir o resultado
        cv.imshow("Mask - White Detection", mask)

        # Sair do loop se a tecla 'q' for pressionada
        if cv.waitKey(1) & 0xFF == ord('q'):
            break

    # Liberar os recursos
    video.release()
    cv.destroyAllWindows()

if __name__ == "__main__":
    main()
