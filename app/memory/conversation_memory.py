class ConversationMemory:
    def __init__(self, max_turns: int = 5):
        self.history = []  # list of {"question": ..., "answer": ...}
        self.max_turns = max_turns

    def add_turn(self, question: str, answer: str):
        self.history.append({"question": question, "answer": answer})
        if len(self.history) > self.max_turns:
            self.history.pop(0)

    def get_history_text(self) -> str:
        if not self.history:
            return ""
        turns = []
        for turn in self.history:
            turns.append(f"User: {turn['question']}\nAssistant: {turn['answer']}")
        return "\n\n".join(turns)

    def clear(self):
        self.history = []