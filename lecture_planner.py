import streamlit as st
import httpx
from dotenv import load_dotenv
from curriculum import get_curriculum_data
from firebase_utils import get_resources_by_topic

OPENROUTER_API_KEY = st.secrets["OPENROUTER_API_KEY"]
PHI_MODEL = "microsoft/phi-4"

def call_phi4(prompt, system_msg, max_tokens=1024):
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "HTTP-Referer": "https://fyp-teacher.local",
        "X-Title": "LecturePlanner"
    }

    data = {
        "model": PHI_MODEL,
        "messages": [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.5,
        "max_tokens": max_tokens
    }

    response = httpx.post("https://openrouter.ai/api/v1/chat/completions", headers=headers, json=data)

    if response.status_code == 200:
        return response.json()["choices"][0]["message"]["content"]
    else:
        return f"❌ Error {response.status_code}: {response.text}"

def generate_prompt(chapter, topic, clos, bloom, resources):
    clo_str = "\n".join([f"- {c}" for c in clos])
    resource_str = "\n".join([f"- {r['title']}" for r in resources])

    return f"""
You are preparing a lecture plan for:

**Chapter:** {chapter}  
**Topic:** {topic}  
**CLOs:**  
{clo_str}  
**Bloom’s Level:** {bloom}  
**Resources:**  
{resource_str}

Prepare a 4-part 45-minute plan:
1. Introduction
2. Explanation
3. Activity
4. Wrap-up
"""

def lecture_planner_page(tab_key):
    st.title("📚 AI Lecture Planner")

    key_prefix = f"lecture_{tab_key}"
    curriculum = get_curriculum_data()
    chapters = curriculum["chapters"]
    selected_chapter = st.selectbox("📘 Select Chapter", list(chapters.keys()), key=f"{key_prefix}_chapter")
    topics = chapters[selected_chapter].get("topics", {})
    selected_topic = st.selectbox("🔍 Select Topic", list(topics.keys()), key=f"{key_prefix}_topic")
    topic_data = topics[selected_topic]
    bloom = topic_data.get("bloom", "Understand")
    topic_clos = topic_data.get("clos", [])
    resources = get_resources_by_topic(selected_topic) or []

    if st.button("✨ Generate AI Lecture Plan", key=f"{key_prefix}_generate"):
        prompt = generate_prompt(selected_chapter, selected_topic, topic_clos, bloom, resources)
        with st.spinner("Generating..."):
            result = call_phi4(prompt, "You are a teaching expert.")
            st.markdown(result)
