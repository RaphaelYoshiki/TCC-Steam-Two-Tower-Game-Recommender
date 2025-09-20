# gui_app.py
import sys
import os
import json
import requests
from typing import Dict, Any

from PySide6.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QHBoxLayout, QTextEdit,
    QLineEdit, QPushButton, QLabel, QMessageBox
)
from PySide6.QtCore import Qt

# ---------------- CONFIG ----------------
RECOMMENDER_RECS = "http://127.0.0.1:8000/recommend"  # endpoint FastAPI
ID_MAPPINGS_PATH = os.path.join("utils", "all_data", "id_mappings.json")
# ----------------------------------------

# --------------- HELPERS ----------------
def load_id_mappings(path: str) -> Dict[str, Dict[str, int]]:
    if not os.path.exists(path):
        return {"genres": {}, "categories": {}, "tags": {}, "developers": {}}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def extract_profile_from_chat(transcript: list) -> Dict[str, Any]:
    """
    Aqui você conecta sua LLM (DeepSeek ou outra) para extrair um perfil
    do histórico de conversa. Exemplo: nomes de gêneros, categorias, etc.
    Neste exemplo simplificado vamos assumir que retorna já formatado.
    """
    # Mock simples (troque pelo seu call LLM)
    return {
        "language": "english",
        "genres": ["Action", "Adventure"],
        "categories": ["Single-player"],
        "tags": ["Multiplayer"],
        "developers": ["Valve"]
    }

def map_names_to_ids(profile: Dict[str, Any], id_maps: Dict[str, Dict[str, int]]) -> Dict[str, list]:
    return {
        "interacted_genres": [id_maps["genres"].get(name, 0) for name in profile.get("genres", [])],
        "interacted_categories": [id_maps["categories"].get(name, 0) for name in profile.get("categories", [])],
        "interacted_tags": [id_maps["tags"].get(name, 0) for name in profile.get("tags", [])],
        "interacted_developers": [id_maps["developers"].get(name, 0) for name in profile.get("developers", [])],
    }

def detect_language_from_transcript(transcript: list) -> str:
    # aqui pode usar heurística ou LLM para detectar idioma
    return "english"
# ----------------------------------------

class ChatWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Recomendador de Jogos - Chat")
        self.resize(820, 640)

        self.id_maps = load_id_mappings(ID_MAPPINGS_PATH)
        self.transcript = [
            {"role": "system", "content": "Você é um assistente que ajuda a recomendar jogos Steam."}
        ]

        layout = QVBoxLayout(self)

        self.chat_display = QTextEdit(self)
        self.chat_display.setReadOnly(True)
        layout.addWidget(self.chat_display)

        input_layout = QHBoxLayout()
        self.user_input = QLineEdit(self)
        self.send_button = QPushButton("Enviar", self)
        self.send_button.clicked.connect(self.send_message)
        input_layout.addWidget(self.user_input)
        input_layout.addWidget(self.send_button)
        layout.addLayout(input_layout)

        self.recommend_button = QPushButton("Gerar Recomendações", self)
        self.recommend_button.clicked.connect(self._build_profile_and_recommend)
        layout.addWidget(self.recommend_button)

        self.recs_display = QTextEdit(self)
        self.recs_display.setReadOnly(True)
        layout.addWidget(QLabel("Recomendações:"))
        layout.addWidget(self.recs_display)

    def send_message(self):
        text = self.user_input.text().strip()
        if not text:
            return
        self.append_message("Você", text)
        self.transcript.append({"role": "user", "content": text})
        self.user_input.clear()

        # Aqui poderia chamar a LLM para responder e enriquecer o chat
        bot_reply = "Entendi! Vou usar isso para suas recomendações."
        self.transcript.append({"role": "assistant", "content": bot_reply})
        self.append_message("Assistente", bot_reply)

    def append_message(self, speaker: str, message: str):
        self.chat_display.append(f"<b>{speaker}:</b> {message}")

    def _build_profile_and_recommend(self):
        try:
            names_profile = extract_profile_from_chat(self.transcript)
            mapped = map_names_to_ids(names_profile, self.id_maps)
            language = detect_language_from_transcript(self.transcript)

            payload_profile = {
                "language": language,
                "interacted_genres": mapped["interacted_genres"],
                "interacted_categories": mapped["interacted_categories"],
                "interacted_tags": mapped["interacted_tags"],
                "interacted_developers": mapped["interacted_developers"],
                "top_k": 10
            }

            r = requests.post(RECOMMENDER_RECS, json=payload_profile, timeout=120)
            r.raise_for_status()
            recs = r.json().get("recommendations", [])

            if not recs:
                self.recs_display.setPlainText("Nenhuma recomendação encontrada.")
            else:
                text = "\n".join([f"{i+1}. {rec['name']} (score={rec['score']:.3f})" for i, rec in enumerate(recs)])
                self.recs_display.setPlainText(text)

        except Exception as e:
            QMessageBox.critical(self, "Erro", str(e))

if __name__ == "__main__":
    app = QApplication(sys.argv)
    win = ChatWindow()
    win.show()
    sys.exit(app.exec())
