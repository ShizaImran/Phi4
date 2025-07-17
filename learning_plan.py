import os
import pickle
import faiss
import json
import numpy as np
import streamlit as st
import httpx
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv

# === Load Environment ===
load_dotenv()
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
PHI_MODEL = "microsoft/phi-4"

# === Configuration ===
INDEX_FILE = "faiss_index/study_resources.index"
METADATA_FILE = "faiss_index/metadata.pkl"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
TOP_K = 8

# === Chapter-Topic Mapping ===
CHAPTER_TOPICS = {
    "Chapter 1: Binary Systems and Hexadecimal": [
        "Introduction to Binary",
        "Denary to Binary Conversion",
        "Binary to Denary Conversion",
        "Hexadecimal Basics",
        "Binary vs Hexadecimal"
    ],
    "Chapter 2: Communication and Internet Technologies": [
        "Serial & Parallel Transmission",
        "USB & Protocols",
        "HTML, HTTP, Web Browsers",
        "Error Checking Methods"
    ],
    "Chapter 3: Logic Gates and Logic Circuits": [
        "Basic Logic Gates",
        "Truth Tables",
        "Logic Circuits in Real World",
        "XOR, NAND, NOR Applications"
    ],
    "Chapter 4: Operating Systems and Computer Architecture": [
        "Functions of OS",
        "Interrupts & Buffers",
        "Fetch-Execute Cycle"
    ],
    "Chapter 5: Input and Output Devices": [
        "Scanners & Cameras",
        "Printers & Projectors",
        "Sensors & Microphones",
        "Actuators & Touch Screens"
    ],
    "Chapter 6: Memory and Data Storage": [
        "File Formats (JPEG, MP3, etc.)",
        "Lossless vs Lossy Compression",
        "Primary vs Secondary Storage"
    ],
    "Chapter 7: High- and Low-Level Languages": [
        "High-Level vs Low-Level",
        "Compilers vs Interpreters",
        "Syntax & Logic Errors"
    ],
    "Chapter 8: Security and Ethics": [
        "Viruses & Hacking",
        "Encryption & Firewalls",
        "Computer Ethics & Privacy"
    ]
}

# === Load Models and Metadata ===
@st.cache_resource
def load_models():
    model = SentenceTransformer(EMBEDDING_MODEL)
    index = faiss.read_index(INDEX_FILE)
    return model, index

@st.cache_data
def load_metadata():
    with open(METADATA_FILE, "rb") as f:
        return pickle.load(f)

model, index = load_models()
metadata = load_metadata()

# === Recommendation Engine ===
def retrieve_resources(query: str, k=TOP_K):
    query_vec = model.encode([query])
    D, I = index.search(np.array(query_vec).astype("float32"), k)
    return [metadata[i] for i in I[0] if i < len(metadata)]

# === Phi-4 API Call ===
def call_phi4(prompt, system_msg, max_tokens=1200):
    try:
        headers = {
            "Authorization": f"Bearer {OPENROUTER_API_KEY}",
            "HTTP-Referer": "https://fyp-student.local",
            "X-Title": "LearningPlanGenerator"
        }

        data = {
            "model": PHI_MODEL,
            "messages": [
                {"role": "system", "content": system_msg},
                {"role": "user", "content": prompt}
            ],
            "temperature": 0.6,
            "max_tokens": max_tokens
        }

        response = httpx.post("https://openrouter.ai/api/v1/chat/completions", headers=headers, json=data)

        if response.status_code == 200:
            return response.json()["choices"][0]["message"]["content"]
        else:
            raise Exception(f"❌ Phi API Error {response.status_code}: {response.text}")

    except Exception as e:
        print("❌ API Call Failed:", e)
        return None

# === Prompt ===
def generate_study_plan_prompt(style, chapter, topic, resources):
    res_text = "\n".join([f"- {r.get('title', 'Untitled')}: {r.get('description', '')}" for r in resources])
    return f"""
You are a Computer Science teacher generating a personalized student study plan.

**Chapter:** {chapter}
**Topic:** {topic}
**Learning Style:** {style}

**Available Resources:**  
{res_text}

🎓 Instructions:
1. Create a 5-step learning plan
2. Include both theory and practice
3. Suggest assessments or self-checks
4. Format clearly using markdown and emojis
5. Keep it relevant to the topic only
"""

# === Streamlit UI ===
def learning_plan_page():
    st.title("📚 Personalized Learning Plan Generator")

    selected_chapter = st.selectbox("📘 Select Chapter", list(CHAPTER_TOPICS.keys()))
    selected_topic = st.selectbox("🔍 Select Topic", CHAPTER_TOPICS[selected_chapter])
    learning_style = st.selectbox("🎨 Learning Style", ["Visual", "Auditory", "Reading/Writing", "Kinesthetic"])

    if st.button("✨ Generate Learning Plan", type="primary"):
        with st.spinner("Generating your plan..."):
            search_query = f"{selected_chapter}: {selected_topic}"
            resources = retrieve_resources(search_query)

            prompt = generate_study_plan_prompt(learning_style, selected_chapter, selected_topic, resources)
            plan = call_phi4(prompt, system_msg="You are a computer science education expert generating personalized study plans.")

            if plan:
                st.subheader("📖 Generated Study Plan")
                st.markdown(plan)

                st.subheader("📚 Resources")
                for i, res in enumerate(resources[:5], 1):
                    st.markdown(f"**{i}. [{res.get('title', 'Untitled')}]({res.get('url', '#')})**")
                    st.caption(res.get("description", ""))

            else:
                st.error("⚠️ Failed to generate plan. Please try again.")

if __name__ == "__main__":
    learning_plan_page()
