import cv2 as cv
import numpy as np

def roi_bottom_half(frame):
    height, width = frame.shape[:2]
    return frame[height // 2:, :]

def process_frame(frame):
    frame = roi_bottom_half(frame)
    # Converter o frame de BGR para HSV
    hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)

    lower_white = np.array([20,9,220])   #0,0,200
    upper_white = np.array([180,45,255]) # 180,55,255 azul neon
    
    mask = cv.inRange(hsv, lower_white, upper_white)

    # Aplicar a máscara para extrair a cor branca da imagem original
    output = cv.bitwise_and(frame, frame, mask=mask)
    blurred = cv.GaussianBlur(output, (7, 7), 0)
    edges= cv.Canny(blurred, 75,150)
    cv.imshow("Mask", mask)

    length = 5 
    gap = 9      
    lines = cv.HoughLinesP(edges, 1, np.pi / 180, 50, minLineLength=length, maxLineGap=gap)

    if lines is not None:
        for line in lines:
            x1,y1,x2,y2=line[0]
            cv.line(frame,(x1,y1),(x2,y2),(0,255,0),3)
            output = cv.bitwise_and(frame, frame, mask=mask)

    return output, mask

def main():
    video = cv.VideoCapture("/Users/sofialinheira/Desktop/IC/codigos_teste/videos/lane_video.mp4")

    while True:
        ret, frame = video.read()

        
        if not ret:
            break

        output, mask = process_frame(frame)

        # Exibir o resultado
        cv.imshow("Mask - White Detection", frame)
        

        # Sair do loop se a tecla 'q' for pressionada
        if cv.waitKey(1) & 0xFF == ord('q'):
            break

    # Liberar os recursos
    video.release()
    cv.destroyAllWindows()

if __name__ == "__main__":
    main()
