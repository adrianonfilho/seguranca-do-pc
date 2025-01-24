import cv2
import requests
import os
import time

# Configurações do Telegram
TELEGRAM_BOT_TOKEN = ''  # Substitua pelo token do seu bot
TELEGRAM_CHAT_ID = ''  # Substitua pelo ID do chat

def enviar_mensagem_telegram(mensagem):
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    params = {
        'chat_id': TELEGRAM_CHAT_ID,
        'text': mensagem
    }
    response = requests.post(url, params=params)
    return response.json()

def enviar_foto_telegram(caminho_foto):
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendPhoto"
    with open(caminho_foto, 'rb') as foto:
        files = {'photo': foto}
        params = {'chat_id': TELEGRAM_CHAT_ID}
        response = requests.post(url, files=files, params=params)
    return response.json()

# Carrega o modelo Haar Cascade para detecção de rostos
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Verifica se o modelo foi carregado corretamente
if face_cascade.empty():
    print("Erro: Não foi possível carregar o modelo Haar Cascade.")
    exit()

# Inicializa a webcam
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

if not cap.isOpened():
    print("Erro: Não foi possível abrir a câmera.")
    exit()

print("Câmera aberta com sucesso! Pressione 'q' para sair.")

# Variável para armazenar o tempo do último envio
ultimo_envio = 0
# Intervalo entre os envios (em segundos)
intervalo_envio = 7

while True:
    # Captura um frame da câmera
    ret, frame = cap.read()
    if not ret:
        print("Erro: Não foi possível capturar o frame.")
        break

    # Converte o frame para escala de cinza (o Haar Cascade funciona melhor com imagens em preto e branco)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detecta rostos no frame
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    if len(faces) > 0 and (time.time() - ultimo_envio) > intervalo_envio:
        print("Rosto detectado!")

        # Salva o frame como uma imagem temporária
        caminho_foto = "rosto_detectado.jpg"
        cv2.imwrite(caminho_foto, frame)

        # Envia a mensagem e a foto para o Telegram
        enviar_mensagem_telegram("Alguém está na frente do computador!")
        enviar_foto_telegram(caminho_foto)

        # Remove a imagem temporária após o envio
        os.remove(caminho_foto)

        ultimo_envio = time.time()

    # Desenha retângulos ao redor dos rostos detectados
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)  

    # Exibe o frame com as detecções
    cv2.imshow('Vigilante', frame)

    # Pressione 'q' para sair do loop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Libera a câmera e fecha as janelas
cap.release()
cv2.destroyAllWindows()
