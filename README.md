# 🤖 UPES Buddy

**UPES Buddy** is an intelligent chatbot powered by Large Language Models (LLMs), built using Python and integrated with Telegram via webhook. It is designed to provide real-time conversational support for students, faculty, and visitors of the University of Petroleum and Energy Studies (UPES), answering queries related to academics, events, facilities, and more.

---

## 📌 Table of Contents

- [Features](#features)  
- [Tech Stack](#tech-stack)  
- [Usage](#usage)  
- [LLM Workflow](#llm-workflow)  
- [Security](#security)  
- [Future Scope](#future-scope)  
- [Contributors](#contributors)

---

## 🔍 Features

- 🎓 **University-Specific Assistance**: Handles queries related to academics, courses, schedules, and events at UPES.
- 🤖 **LLM Integration**: Leverages OpenAI's GPT or similar models for generating intelligent responses.
- 📄 **RAG Support**: Uses Retrieval-Augmented Generation to fetch context from university documents.
- 💬 **Telegram Bot**: Fully integrated with Telegram using webhooks for instant replies.
- 🔁 **Multi-turn Conversation**: Maintains context to handle follow-up questions (optional).
- 🧠 **Smart Document Search**: Finds and fetches relevant data using vector embeddings.

---

## 🛠️ Tech Stack

| Component        | Technology                     |
|------------------|--------------------------------|
| Backend          | Python (Flask / FastAPI)       |
| LLM Integration  | OpenAI GPT / LangChain         |
| Document Store   | FAISS / Chroma / Pinecone      |
| Bot Integration  | python-telegram-bot / Telebot  |
| Deployment       | Ngrok / Render / Railway       |
| Environment Vars | Python-dotenv                  |
| Voice (Optional) | OpenAI Whisper API             |

---

## ⚙️ Usage

- **Start a Conversation**: Open the Telegram bot (e.g., [@UPESBuddyBot](https://t.me/YourBotUsername)) and type your question.
- **Ask Questions**: About academic schedules, hostel info, faculty contacts, or events.
- **Bot Responds**: Using LLM + university document knowledge base.

---

## 🧠 LLM Workflow

```text
[User Input] ──▶ Telegram Webhook
                    │
            [Parse Message]
                    │
        [Retrieve Relevant Docs] ◀─ (Optional: Chroma/FAISS)
                    │
       [Query + Context to LLM (OpenAI)]
                    │
           [Generate + Send Response]
                    ▼
              [Telegram User]
```

---

## 🔐 Security

- Use `.env` to store and load all secrets (API keys, DB credentials, tokens).
- Do not expose API keys publicly.
- Implement rate limiting or CAPTCHA if public-facing.

---

## 📈 Future Scope

- 📢 Voice Input and Text-to-Speech responses
- 📅 Smart academic calendar integration
- 📊 Admin analytics dashboard for query stats
- 🤝 WhatsApp or Web App version
- 🧠 Custom fine-tuned model trained on UPES FAQs

---

## 👨‍💻 Contributors

- **Konal Puri** – Lead Developer  

> Proudly built to empower the UPES community using LLMs 🤍

---
