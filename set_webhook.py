import requests

BOT_TOKEN = "8026614347:AAFBeiQu__k2iu4svmqq2R7RcWIm5Y3XTjs"
WEBHOOK_URL = "https://307c-103-131-60-82.ngrok-free.app/webhook"  # Replace with your ngrok URL

response = requests.get(
    f"https://api.telegram.org/bot{BOT_TOKEN}/setWebhook?url={WEBHOOK_URL}"
)

if response.status_code == 200:
    print("Webhook set successfully!")
else:
    print("Failed to set webhook:", response.text)
