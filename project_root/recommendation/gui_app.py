# gui_app_updated.py
import sys
import os
import json
import requests
from typing import Dict, Any, List, Optional

from PySide6.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QHBoxLayout, QTextEdit,
    QLineEdit, QPushButton, QLabel, QMessageBox, QInputDialog
)
from PySide6.QtCore import Qt

# ---------------- CONFIG ----------------
# Endereço do serviço FastAPI (seu backend)
BACKEND_BASE = os.environ.get("RECOMMENDER_BACKEND", "http://127.0.0.1:8000")
RECOMMEND_ENDPOINT = f"{BACKEND_BASE}/recommend"
CHECK_PROFILE_ENDPOINT = f"{BACKEND_BASE}/check_profile"
PROFILE_FROM_CHAT_ENDPOINT = f"{BACKEND_BASE}/profile_from_chat"

# Endereço da API do DeepSeek (LLM) — configure conforme sua API
DEEPSEEK_API = os.environ.get("DEEPSEEK_API", "https://api.deepseek.com")  # exemplo
DEEPSEEK_API_KEY = os.environ.get("DEEPSEEK_API_KEY", "sk-e71eacef903346a89e9b029c4bb08551")

# Caminho para mapeamentos nome->id (opcional, usado para converter rótulos)
ID_MAPPINGS_PATH = os.path.join("utils", "all_data", "id_mappings.json")
# ----------------------------------------

# --------------- HELPERS ----------------
def load_id_mappings(path: str) -> Dict[str, Dict[str, int]]:
    if not os.path.exists(path):
        return {"genres": {}, "categories": {}, "tags": {}, "developers": {}}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def map_labels_to_ids(maybe_profile: dict, id_maps: dict) -> dict:
    """
    Converte rótulos (strings) para ids (ints) usando id_maps quando necessário.
    O deepseek pode enviar 'top_genre' como 'RPG' em vez de id; convertemos quando possível.
    """
    out = dict(maybe_profile)  # shallow copy
    # mapping for single fields top_genre/top_cat/top_tag if strings
    label_fields = {
        "top_genre": ("genres",),
        "top_cat": ("categories",),
        "top_tag": ("tags",),
    }
    for field, (map_key,) in label_fields.items():
        if field in out and isinstance(out[field], str):
            mapped = id_maps.get(map_key, {}).get(out[field])
            if mapped is not None:
                out[field] = int(mapped)
            else:
                # try case-insensitive match
                for k, v in id_maps.get(map_key, {}).items():
                    if k.lower() == out[field].lower():
                        out[field] = int(v)
                        break
    # candidate_appids, if labels present (rare), ignore conversion
    # genre_dominance / diversity ensure numeric
    if "genre_dominance" in out:
        try:
            out["genre_dominance"] = float(out["genre_dominance"])
        except Exception:
            pass
    if "genre_diversity" in out:
        try:
            out["genre_diversity"] = float(out["genre_diversity"])
        except Exception:
            pass
    return out

def try_parse_json(text: str) -> Optional[dict]:
    """
    Tenta extrair JSON do texto. Aceita duas situações:
    - A resposta inteira é JSON
    - A resposta tem um bloco JSON (tenta encontrar o primeiro '{' ... matching)
    """
    text = text.strip()
    # caso simples: JSON puro
    try:
        return json.loads(text)
    except Exception:
        pass
    # procurar primeiro '{' e tentar loads progressivamente até achar um válido
    first = text.find('{')
    if first == -1:
        return None
    # tentar extrair substring que termina no final
    for end in range(len(text), first, -1):
        snippet = text[first:end]
        try:
            return json.loads(snippet)
        except Exception:
            continue
    return None

def extract_profile_from_chat_local(transcript: List[dict], id_maps: Dict[str, Dict[str, int]]) -> dict:
    """
    Fallback simples: tenta extrair gênero/categorias com heurística local e mapear para ids
    - Usa palavras-chaves no transcript para formar um perfil básico.
    É apenas um fallback para testes; idealmente o DeepSeek gera o JSON final.
    """
    joined = " ".join([m["content"] for m in transcript if m["role"] in ("user", "assistant")]).lower()
    profile = {}
    # heurísticas simples (você pode aumentar)
    if "rpg" in joined:
        profile["top_genre"] = id_maps.get("genres", {}).get("RPG", 0)
    elif "action" in joined or "ação" in joined:
        profile["top_genre"] = id_maps.get("genres", {}).get("Action", 0)
    # defaults when unknown (0)
    profile.setdefault("top_genre", 0)
    profile.setdefault("top_cat", 0)
    profile.setdefault("top_tag", 0)
    profile.setdefault("lang_id", 0)
    profile.setdefault("genre_dominance", 0.5)
    profile.setdefault("genre_diversity", 1.0)
    return profile

def post_to_deepseek(messages: List[dict]) -> str:
    """
    Chama a API do DeepSeek. Usa o cliente OpenAI (OpenAI python package)
    quando DEEPSEEK_API_KEY estiver presente; caso contrário, tenta requests.post
    para a URL em DEEPSEEK_API (modo local).
    """
    # preferir usar o cliente OpenAI se chave presente
    if DEEPSEEK_API_KEY:
        try:
            from openai import OpenAI
        except Exception as e:
            return f"[Erro: pacote openai não instalado: {e}]"

        try:
            client = OpenAI(api_key=DEEPSEEK_API_KEY, base_url=DEEPSEEK_API)
            response = client.chat.completions.create(
                model="deepseek-chat",
                messages=messages,
                stream=False
            )
            # resposta esperada conforme doc: response.choices[0].message.content
            try:
                return response.choices[0].message.content
            except Exception:
                # fallback para tentar decodificar dicts
                try:
                    return response["choices"][0]["message"]["content"]
                except Exception:
                    return str(response)
        except Exception as e:
            return f"[Erro ao chamar DeepSeek (cliente OpenAI): {e}]"

    # fallback: usar HTTP direto para um servidor DeepSeek local (se estiver rodando)
    # DEEPSEEK_API nesse caso deve ser algo como "http://127.0.0.1:9000/ask"
    try:
        headers = {"Content-Type": "application/json"}
        payload = {"messages": messages}
        r = requests.post(DEEPSEEK_API, json=payload, headers=headers, timeout=30)
        r.raise_for_status()
        # tentar parse de JSON
        try:
            data = r.json()
        except Exception:
            return r.text
        # formatos possíveis: { "content": "..."} | {"choices":[{"message": {"content": "..."}}]}
        if isinstance(data, dict):
            if "content" in data and isinstance(data["content"], str):
                return data["content"]
            if "choices" in data and isinstance(data["choices"], list) and data["choices"]:
                ch = data["choices"][0]
                if isinstance(ch, dict):
                    if "message" in ch and isinstance(ch["message"], dict) and "content" in ch["message"]:
                        return ch["message"]["content"]
                    if "text" in ch:
                        return ch["text"]
        return str(data)
    except Exception as e:
        return f"[Erro ao chamar DeepSeek via HTTP: {e}]"

# ----------------------------------------

class ChatWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Recomendador de Jogos - Chat (DeepSeek)")
        self.resize(900, 700)

        self.id_maps = load_id_mappings(ID_MAPPINGS_PATH)
        self.transcript: List[dict] = [
            {"role": "system", "content": (
                "Você é um assistente que coleta informações para gerar um perfil de usuário "
                "para um sistema de recomendação de jogos. "
                "Pergunte ao usuário até ter os campos: lang_id, top_genre, top_cat, top_tag, "
                "genre_dominance (0.0-1.0) e genre_diversity (número). "
                "Quando tiver o perfil completo, retorne SOMENTE o JSON com esses campos (ids ou rótulos) "
                "sem texto adicional."
            )}
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

        control_layout = QHBoxLayout()
        self.recommend_button = QPushButton("Gerar Recomendações (force)", self)
        self.recommend_button.clicked.connect(self._force_build_profile_and_recommend)
        control_layout.addWidget(self.recommend_button)

        self.check_profile_button = QPushButton("Checar profile atual", self)
        self.check_profile_button.clicked.connect(self._check_profile_current)
        control_layout.addWidget(self.check_profile_button)

        self.clear_button = QPushButton("Limpar Chat", self)
        self.clear_button.clicked.connect(self._clear_chat)
        control_layout.addWidget(self.clear_button)

        layout.addLayout(control_layout)

        self.recs_display = QTextEdit(self)
        self.recs_display.setReadOnly(True)
        layout.addWidget(QLabel("Recomendações:"))
        layout.addWidget(self.recs_display)

        # show initial system message
        #self.append_message("system", self.transcript[0]["content"])

    def append_message(self, speaker: str, message: str):
        if speaker == "system":
            self.chat_display.append(f"<i><small>{message}</small></i>")
        elif speaker == "assistant":
            self.chat_display.append(f"<b>Assistente:</b> {message}")
        elif speaker == "user":
            self.chat_display.append(f"<b>Você:</b> {message}")
        else:
            self.chat_display.append(f"<b>{speaker}:</b> {message}")

    def send_message(self):
        text = self.user_input.text().strip()
        if not text:
            return
        # append user message locally
        self.append_message("Você", text)
        self.transcript.append({"role": "user", "content": text})
        self.user_input.clear()

        # call DeepSeek with the full transcript so it can continue collecting profile
        assistant_text = post_to_deepseek(self.transcript)

        # append assistant reply
        self.transcript.append({"role": "assistant", "content": assistant_text})
        # try to parse JSON profile
        parsed = try_parse_json(assistant_text)
        if parsed:
            # show neatly the JSON (for debug)
            self.append_message("assistant", f"[JSON recebido]\n<pre>{json.dumps(parsed, ensure_ascii=False, indent=2)}</pre>")
            # convert labels -> ids if needed
            mapped = map_labels_to_ids(parsed, self.id_maps)
            # validate with backend /check_profile
            try:
                r = requests.post(CHECK_PROFILE_ENDPOINT, json=mapped, timeout=10)
                r.raise_for_status()
                check = r.json()
            except Exception as e:
                QMessageBox.warning(self, "Aviso", f"Erro ao checar perfil: {e}")
                return
            if not check.get("all_good", False):
                missing = check.get("missing_fields", [])
                invalid = check.get("invalid_fields", {})
                msg = "Perfil incompleto/inválido:\n"
                if missing:
                    msg += f"Campos faltando: {missing}\n"
                if invalid:
                    msg += f"Campos inválidos: {invalid}\n"
                # Inform the user and offer to answer missing fields manually
                QMessageBox.information(self, "Perfil incompleto", msg)
                # let the assistant continue the conversation (we still keep transcript)
            else:
                # profile ok -> call profile_from_chat endpoint
                try:
                    r2 = requests.post(PROFILE_FROM_CHAT_ENDPOINT, json=mapped, timeout=30)
                    r2.raise_for_status()
                    reco_json = r2.json()
                    self._display_recommendations_recojson(reco_json)
                except Exception as e:
                    QMessageBox.critical(self, "Erro", f"Erro ao obter recomendações: {e}")
        else:
            # normal assistant text — display as usual
            self.append_message("Assistente", assistant_text)

    def _check_profile_current(self):
        """
        Constrói um profile parcial usando heurística local (fallback) e checa no backend.
        """
        try:
            partial = extract_profile_from_chat_local(self.transcript, self.id_maps)
            mapped = map_labels_to_ids(partial, self.id_maps)
            r = requests.post(CHECK_PROFILE_ENDPOINT, json=mapped, timeout=10)
            r.raise_for_status()
            check = r.json()
            QMessageBox.information(self, "Check Profile", json.dumps(check, ensure_ascii=False, indent=2))
        except Exception as e:
            QMessageBox.critical(self, "Erro", str(e))

    def _force_build_profile_and_recommend(self):
        """
        Fallback manual: tenta extrair perfil localmente, pede ao usuário confirmar/editar os campos,
        e então chama /profile_from_chat.
        """
        try:
            partial = extract_profile_from_chat_local(self.transcript, self.id_maps)
            # ask user to confirm/edit fields
            editable = dict(partial)
            fields = ["lang_id", "top_genre", "top_cat", "top_tag", "genre_dominance", "genre_diversity"]
            for f in fields:
                current = editable.get(f, "")
                val, ok = QInputDialog.getText(self, f"Confirmar {f}", f"Valor atual para {f}:", text=str(current))
                if not ok:
                    QMessageBox.information(self, "Cancelado", "Operação cancelada pelo usuário.")
                    return
                # try cast
                try:
                    if f in ["lang_id", "top_genre", "top_cat", "top_tag"]:
                        editable[f] = int(val)
                    else:
                        editable[f] = float(val)
                except Exception:
                    editable[f] = val
            # map labels -> ids if necessary
            mapped = map_labels_to_ids(editable, self.id_maps)
            # validate
            r = requests.post(CHECK_PROFILE_ENDPOINT, json=mapped, timeout=10)
            r.raise_for_status()
            check = r.json()
            if not check.get("all_good", False):
                QMessageBox.warning(self, "Perfil inválido", json.dumps(check, ensure_ascii=False, indent=2))
                return
            # call profile_from_chat
            r2 = requests.post(PROFILE_FROM_CHAT_ENDPOINT, json=mapped, timeout=30)
            r2.raise_for_status()
            reco_json = r2.json()
            self._display_recommendations_recojson(reco_json)
        except Exception as e:
            QMessageBox.critical(self, "Erro", str(e))

    def _display_recommendations_recojson(self, reco_json: dict):
        """
        Espera o schema RecoResponse { user_profile, top_k, results: [{appid, score, title}] }
        e formata para exibição.
        """
        try:
            results = reco_json.get("results", [])
            if not results:
                self.recs_display.setPlainText("Nenhuma recomendação retornada.")
                return
            lines = []
            for i, row in enumerate(results):
                title = row.get("title") or row.get("name") or str(row.get("appid"))
                score = row.get("score", None)
                if score is not None:
                    lines.append(f"{i+1}. {title}  (score={float(score):.3f})")
                else:
                    lines.append(f"{i+1}. {title}")
            self.recs_display.setPlainText("\n".join(lines))
            # append a short assistant-style message to chat
            self.append_message("assistant", f"Encontrei {len(results)} recomendação(ões) para você. Veja abaixo.")
        except Exception as e:
            QMessageBox.critical(self, "Erro", f"Erro ao formatar recomendações: {e}")

    def _clear_chat(self):
        self.chat_display.clear()
        self.recs_display.clear()
        self.transcript = [self.transcript[0]]  # keep system
        self.append_message("system", self.transcript[0]["content"])


# ----------------------------------------

if __name__ == "__main__":
    app = QApplication(sys.argv)
    win = ChatWindow()
    win.show()
    sys.exit(app.exec())
