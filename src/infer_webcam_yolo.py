"""Roda o YOLOv8 fine-tunado na webcam do PC, em tempo real.

Abre uma janela mostrando o vídeo da câmera com as caixas (bounding boxes)
desenhadas em cima dos objetos detectados. Aperte 'q' para fechar.
"""

from pathlib import Path

import cv2
from ultralytics import YOLO

# Caminhos: usamos os mesmos pesos treinados ('best.pt') dos outros scripts.
ROOT = Path(__file__).resolve().parent.parent
WEIGHTS = ROOT / "runs" / "detect" / "train-2" / "weights" / "best.pt"

# Índice da câmera. 0 = webcam padrão. Se você tiver mais de uma câmera,
# troque para 1, 2, etc. até achar a certa.
CAMERA_INDEX = 0

# Confiança mínima para considerar uma detecção (0 a 1).
# Mais alto = menos caixas, porém mais certeiras.
CONF = 0.4


def main():
    # 1) Carrega o modelo treinado uma única vez (fora do loop).
    model = YOLO(str(WEIGHTS))

    # 2) Abre a webcam. cv2.CAP_AVFOUNDATION é o backend nativo do macOS
    #    e costuma abrir a câmera mais rápido/sem travar.
    cap = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_AVFOUNDATION)
    if not cap.isOpened():
        raise RuntimeError(
            f"Não consegui abrir a câmera {CAMERA_INDEX}. "
            "Verifique se outro app está usando a webcam ou tente outro índice."
        )

    print("Webcam aberta. Aperte 'q' na janela do vídeo para sair.")

    # 3) Loop principal: lê um frame, roda o YOLO, mostra o resultado.
    while True:
        # ret = False quando a leitura falha (câmera desconectada, fim, etc.).
        ret, frame = cap.read()
        if not ret:
            print("Falha ao ler frame da câmera. Encerrando.")
            break

        # Roda a detecção neste frame.
        #   stream=False -> retorna a lista de resultados deste frame
        #   verbose=False -> não polui o terminal a cada frame
        results = model.predict(frame, conf=CONF, verbose=False)

        # results[0].plot() devolve o MESMO frame, já com as caixas,
        # rótulos e confiança desenhados em cima (imagem em formato BGR,
        # que é o que o OpenCV espera para exibir).
        annotated = results[0].plot()

        # Quantos objetos foram detectados neste frame (opcional, só informativo).
        n = 0 if results[0].boxes is None else len(results[0].boxes)
        cv2.putText(
            annotated,
            f"deteccoes: {n}",
            (10, 30),                      # posição (x, y) do texto
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,                           # tamanho da fonte
            (0, 255, 0),                   # cor (verde, em BGR)
            2,                             # espessura
        )

        # Mostra o frame anotado numa janela.
        cv2.imshow("YOLO - Webcam (aperte 'q' para sair)", annotated)

        # Espera 1ms por uma tecla. Se for 'q', sai do loop.
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    # 4) Limpeza: libera a câmera e fecha as janelas.
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
