import os
from groq import Groq
from dotenv import load_dotenv

load_dotenv()
client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

def ask_groq(question: str, context_chunks: list[dict], model: str = "openai/gpt-oss-20b") -> str:
    context = "\n\n".join(
        f"[Page {c['page_number']}]: {c['text']}" for c in context_chunks
    )

    prompt = f"""Answer the question using ONLY the context below. If the answer isn't in the context, say you don't know.

Context:
{context}

Question: {question}

Answer:"""

    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,
    )
    return response.choices[0].message.content



def rewrite_question(question: str, history_text: str, model: str = "openai/gpt-oss-20b") -> str:
    if not history_text:
        return question  # first question, no history to consider

    prompt = f"""Given the conversation history and a follow-up question, rewrite the follow-up question to be a standalone question that includes all necessary context. Do not answer it, just rewrite it.

Conversation history:
{history_text}

Follow-up question: {question}

Standalone question:"""

    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
    )
    return response.choices[0].message.content.strip()