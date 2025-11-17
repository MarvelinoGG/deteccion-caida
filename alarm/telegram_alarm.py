import requests

TOKEN = "8208139880:AAHu_zV7KlDu3olWbyefDsSyRuIKAI4TlSc"
BASE_URL = f"https://api.telegram.org/bot{TOKEN}"
url_updates = f"https://api.telegram.org/bot{TOKEN}/getUpdates"
url_send_message = f"https://api.telegram.org/bot{TOKEN}/sendMessage"
CHAT_ID = 8480322408  # ID del usuario de telegram
# Coordenadas del dispositivo
LAT = 39.479000
LON = -6.342167

def get_telegram_updates():
    response = requests.get(url_updates)
    print(response.text)
    if response.status_code == 200:
        return response.json()
    else:
        print("Error al obtener actualizaciones:", response.status_code)
        return None
    
def send_telegram_message(text):
    payload = {
        'chat_id': CHAT_ID,
        'text': text
    }
    response = requests.post(url_send_message, data=payload)
    if response.status_code == 200:
        print("Mensaje enviado correctamente")
    else:
        print("Error al enviar mensaje:", response.status_code)


def send_telegram_location(chat_id, lat, lon):
    url = f"{BASE_URL}/sendLocation"
    payload = {
        'chat_id': chat_id,
        'latitude': lat,
        'longitude': lon
    }
    response = requests.post(url, data=payload)
    if response.status_code == 200:
        print("Ubicación enviada correctamente")
    else:
        print("Error al enviar ubicación:", response.status_code, response.text)

def send_fall_alert_with_location():
    # 1) Mensaje con enlace a Google Maps
    maps_url = f"https://www.google.com/maps?q={LAT},{LON}"
    text = (
        "⚠️ Se ha detectado una posible caída.\n"
        f"Ubicación aproximada (por IP):\n{maps_url}"
    )
    send_telegram_message(text)

    # 2) Ubicación nativa de Telegram (mapita dentro del chat)
    send_telegram_location(CHAT_ID, LAT, LON)
