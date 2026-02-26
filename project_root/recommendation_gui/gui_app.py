import json
import toga
from toga.style import Pack
from toga.style.pack import COLUMN, ROW

from deepseek_client import DeepSeekClient
from chat_state import ChatState
from prompts import SYSTEM_PROMPT
from backend_client import fetch_recommendations


class ChatApp(toga.App):

    def startup(self):
        self.client = DeepSeekClient()
        self.state = ChatState(SYSTEM_PROMPT)

        self.chat_log = toga.MultilineTextInput(readonly=True, style=Pack(flex=1))
        self.input_box = toga.TextInput(style=Pack(flex=1))
        self.send_button = toga.Button("Send", on_press=self.send_message)

        input_row = toga.Box(
            children=[self.input_box, self.send_button],
            style=Pack(direction=ROW, padding=5)
        )

        self.main_window = toga.MainWindow(title="Game Recommender Chat")
        self.main_window.content = toga.Box(
            children=[self.chat_log, input_row],
            style=Pack(direction=COLUMN)
        )
        self.main_window.show()

    def send_message(self, widget):
        user_text = self.input_box.value.strip()
        if not user_text:
            return

        self.append(f"You: {user_text}\n")
        self.input_box.value = ""
        self.state.add_user(user_text)

        msg = self.client.chat(self.state.get())

        # ================================
        # CASO 1: DeepSeek chamou a tool
        # ================================
        print("MSG:", msg)
        print("TOOL CALLS:", msg.tool_calls)
        
        if msg.tool_calls:
            
            self.state.messages.append({
                "role": "assistant",
                "tool_calls": [
                    {
                        "id": msg.tool_calls[0].id,
                        "type": "function",
                        "function": {
                            "name": msg.tool_calls[0].function.name,
                            "arguments": msg.tool_calls[0].function.arguments
                        }
                    }
                ],
                "content": None
            })
            
            tool_call = msg.tool_calls[0]
            args = json.loads(tool_call.function.arguments)

            self.append("🔍 Gerando recomendações...\n")

            profile = {
                "lang_id": args["lang_id"],
                "top_genre": args["top_genre"],
                "top_cat": args["top_cat"],
                "top_tag": args["top_tag"],
                "genre_dominance": args["genre_dominance"],
                "genre_diversity": args["genre_diversity"],
            }

            recs = fetch_recommendations(profile)
            print("RECS RAW:", recs)

            # Retorna o resultado da tool para a LLM
            tool_payload = {
                "recommendations": recs,
                "instruction": (
                    "These are the ONLY games you are allowed to recommend. "
                    "Do not modify this list."
                )
            }
            
            self.state.add_tool_result(
                tool_call.id,
                json.dumps(tool_payload)
            )

            final_msg = self.client.chat(self.state.get())

            self.state.add_assistant(final_msg.content)
            self.append(f"Assistant: {final_msg.content}\n\n")

        # ================================
        # CASO 2: conversa normal
        # ================================
        else:
            self.state.add_assistant(msg.content)
            self.append(f"Assistant: {msg.content}\n\n")

    def append(self, text):
        self.chat_log.value += text


def main():
    return ChatApp(
        formal_name="Game Recommender Chat",
        app_id="br.com.tcc.gamerecommender"
    )

if __name__ == "__main__":
    main().main_loop()
