class ChatState:
    def __init__(self, system_prompt):
        self.messages = [
            {"role": "system", "content": system_prompt}
        ]

    def add_user(self, text):
        self.messages.append({"role": "user", "content": text})

    def add_assistant(self, text):
        self.messages.append({"role": "assistant", "content": text})

    def add_tool_result(self, tool_call_id, result):
        self.messages.append({
            "role": "tool",
            "tool_call_id": tool_call_id,
            "content": str(result)
        })

    def get(self):
        return self.messages
