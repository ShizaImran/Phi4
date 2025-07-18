import os
import json
import httpx
import traceback
from dotenv import load_dotenv
from utils.rag_utils import retrieve_context

#load_dotenv()
OPENROUTER_API_KEY = st.secrets["OPENROUTER_API_KEY"]
PHI_MODEL = "microsoft/phi-4"

# === 🔁 Universal Phi-4 Call ===
def call_phi4(prompt, system_msg, max_tokens=1024):
    try:
        headers = {
            "Authorization": f"Bearer {OPENROUTER_API_KEY}",
            "HTTP-Referer": "https://fyp-study.local",  # customize this
            "X-Title": "FYP-QuizEngine"
        }

        payload = {
            "model": PHI_MODEL,
            "messages": [
                {"role": "system", "content": system_msg},
                {"role": "user", "content": prompt}
            ],
            "temperature": 0.7,
            "max_tokens": max_tokens
        }

        response = httpx.post("https://openrouter.ai/api/v1/chat/completions", headers=headers, json=payload)

        if response.status_code == 200:
            return response.json()["choices"][0]["message"]["content"]
        else:
            raise Exception(f"❌ API Error {response.status_code}: {response.text}")

    except Exception as e:
        print("❌ Phi-4 Call Failed:", e)
        return None

# ✅🟡 INITIAL QUIZ (Profile Setup)
def generate_initial_quiz(profile):
    try:
        level = profile["class_level"]
        style = profile["learning_style"]
        domain = profile["domain"]
        topic = profile["topic"]

        prompt = f"""
You are a smart AI that creates personalized student quizzes.

Student Profile:
- Level: {level}
- Learning Style: {style}
- Domain: {domain}
- Topic: {topic}

🧠 Instructions:
- Generate 10 MCQs for this student.
- Each question must have 4 options (A, B, C, D).
- Return only VALID JSON in this format:
- Each question object must include a `topic` field reflecting the specific subtopic it's testing.

[
  {{
    "question": "What is AI?",
    "options": ["Artificial Intelligence", "Animal", "Ice", "Apple"],
    "answer": "A",
    "topic": "Introduction to AI"
  }},
  ...
]
No explanations. No markdown. No code fences.
"""
        raw = call_phi4(prompt, "You generate JSON-based initial quizzes.", max_tokens=1024)
        if not raw:
            raise ValueError("Empty response from Phi-4")

        content = raw.strip().replace("```json", "").replace("```", "")
        parsed = json.loads(content)

        if not isinstance(parsed, list) or len(parsed) < 5:
            raise ValueError("❌ Invalid or incomplete quiz received")

        return parsed

    except Exception as e:
        print("❌ Initial Quiz JSON Error:", e)
        return [{
            "question": "Fallback: Quiz Generation failed. What do you want's to do now?",
            "options": ["Brather, if you want your son", "high level wrestle", "Send him 2-3 years Dagestan", "& forgot"],
            "answer": "C",
            "topic": topic
        }]

# ✅🟡 PRACTICE QUIZ (RAG-Based)
def generate_practice_quiz(profile, chapter, topic):
    try:
        level = profile.get("class_level", "Matric")
        domain = profile.get("domain", "Computer Science")

        context = retrieve_context(chapter, topic)

        prompt = f"""
You are an AI that generates student quizzes from textbooks.

🧠 Context:
\"\"\"{context}\"\"\"

🎓 Student Info:
- Class Level: {level}
- Domain: {domain}
- Focus Topic: {topic}

Instructions:
- Create 15 MCQs strictly from the provided context.
- Each question must have 4 options (A, B, C, D).
- Each question object must include:
  - "question"
  - "options"
  - "answer"
  - "topic": "{topic}"

Output must be a valid JSON list:
[
  {{
    "question": "...",
    "options": ["A", "B", "C", "D"],
    "answer": "A",
    "topic": "{topic}"
  }},
  ...
]
Do not include explanations, markdown, or code fences.
"""

        raw = call_phi4(prompt, "You return clean JSON quizzes from textbook context.", max_tokens=2048)
        if not raw:
            raise ValueError("Empty response from Phi-4")

        content = raw.strip().replace("```json", "").replace("```", "")
        parsed = json.loads(content)

        if isinstance(parsed, list) and len(parsed) >= 5:
            return parsed
        else:
            raise ValueError("Response did not contain enough valid questions.")

    except Exception as e:
        print("❌ PRACTICE QUIZ FALLBACK TRIGGERED")
        print(traceback.format_exc())
        return [{
            "question": "⚠ Error: Quiz generation failed. Retry?",
            "options": ["Yes", "No", "Maybe", "Contact Support"],
            "answer": "A",
            "topic": topic
        }]
