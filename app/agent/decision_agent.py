from app.llm.groq_client import client

def classify_question(question: str, model: str = "openai/gpt-oss-20b") -> str:
    prompt = f"""Classify the user's message into exactly one category. Reply with ONLY one word: CASUAL, IN_SCOPE, or OUT_OF_SCOPE. Do not explain, do not add punctuation.

Examples:
Message: "hi"
Category: CASUAL

Message: "thanks a lot"
Category: CASUAL

Message: "What is HashSet in Java?"
Category: IN_SCOPE

Message: "What is the capital of Pakistan?"
Category: OUT_OF_SCOPE

Message: "{question}"
Category:"""

    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
        max_tokens=150,
        reasoning_effort="low",      # kam soche, fast rahe
        reasoning_format="hidden",
    )
    raw_output = response.choices[0].message.content.strip()
       # temporary — hata denge baad mein

    category = raw_output.upper().replace(".", "").replace('"', "")

    for valid in ["CASUAL", "IN_SCOPE", "OUT_OF_SCOPE"]:
        if valid in category:
            return valid

    return "IN_SCOPE"  # safe default


def out_of_scope_reply(question: str, model: str = "openai/gpt-oss-20b") -> str:
    prompt = f"""Answer this question briefly in 2-3 lines using your general knowledge. Then add one short line noting this wasn't found in the uploaded document.

Question: "{question}"

Answer:"""
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3,
        max_tokens=300,              # 150 se badhaya, safety ke liye
        reasoning_effort="low",      # naya
        reasoning_format="hidden",   # naya
    )
    result = response.choices[0].message.content.strip()
    if not result:  # safety net — kabhi bhi khaali na jaaye
        return "I'm not fully certain about this, and it's not covered in the uploaded document. You may want to verify this from a reliable source."
    return result


def casual_reply(question: str, model: str = "openai/gpt-oss-20b") -> str:
    prompt = f"""You are a friendly PDF assistant. Reply naturally and briefly to this casual message. Mention you're ready to help answer questions about the uploaded document.

Message: "{question}"

Reply:"""
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.5,
        max_tokens=150,              # 60 se badhaya
        reasoning_effort="low",
        reasoning_format="hidden",
    )
    result = response.choices[0].message.content.strip()
    if not result:
        return "Hey! I'm ready to help — ask me anything about your uploaded document."
    return result

def is_pdf_answer_sufficient(question: str, pdf_answer: str, model: str = "openai/gpt-oss-20b") -> bool:
    """
    Checks if the PDF-based answer fully addresses the question,
    or if it's incomplete/uncertain (e.g. "the document doesn't mention X").
    """
    prompt = f"""You are checking if an answer is complete or incomplete.

Question: "{question}"
Answer: "{pdf_answer}"

Does the answer fully and confidently address the question? Reply with ONLY one word: SUFFICIENT or INCOMPLETE.

Reply INCOMPLETE if the answer says things like "the document doesn't mention", "not found", "no information", or clearly only partially answers the question.
Reply SUFFICIENT if the answer clearly and confidently addresses the question.

Reply:"""

    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
        max_tokens=150,
        reasoning_effort="low",
        reasoning_format="hidden",
    )
    result = response.choices[0].message.content.strip().upper()
    return "SUFFICIENT" in result