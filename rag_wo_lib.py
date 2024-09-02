from typing import Dict, List

import anthropic
import tiktoken

# Set your Anthropic API key
client = anthropic.Anthropic(api_key="")


class RAGSystem:
    def __init__(
        self, model: str = "claude-3-sonnet-20240229", max_tokens: int = 100000
    ):
        self.model = model
        self.max_tokens = max_tokens
        self.history: List[Dict[str, str]] = []
        self.encoding = tiktoken.encoding_for_model(
            "gpt-4"
        )  # Use GPT-4 encoding as an approximation

    def count_tokens(self, text: str) -> int:
        return len(self.encoding.encode(text))

    def truncate_history(self):
        while self.history and self.count_tokens(str(self.history)) > self.max_tokens:
            self.history.pop(0)

    def process_command(self, command: str, content: str) -> str:
        if command == "/clear":
            self.history.clear()
            return "History cleared."
        elif command == "/history":
            return str(self.history)
        else:
            return f"Unknown command: {command}"

    def generate_response(self, prompt: str) -> str:
        if prompt.startswith("/"):
            command, *content = prompt.split(maxsplit=1)
            return self.process_command(command, content[0] if content else "")

        self.truncate_history()

        messages = self.history + [{"role": "user", "content": prompt}]

        response = client.messages.create(
            model=self.model, max_tokens=1024, messages=messages
        )

        ai_response = response.content[0].text
        self.history.append({"role": "user", "content": prompt})
        self.history.append({"role": "assistant", "content": ai_response})

        return ai_response


# Example usage
rag = RAGSystem()

while True:
    user_input = input("You: ")
    if user_input.lower() == "exit":
        break
    response = rag.generate_response(user_input)
    print(f"Claude: {response}")
