import streamlit as st
import datetime
import os
import httpx
from dotenv import load_dotenv
from curriculum import get_curriculum_data
from firebase_utils import get_resources_by_topic

# Load API key
#load_dotenv()
OPENROUTER_API_KEY = st.secrets["OPENROUTER_API_KEY"]
PHI_MODEL = "microsoft/phi-4"

# === Phi-4 API Call ===
def call_phi4(prompt, system_msg, max_tokens=1024):
    try:
        headers = {
            "Authorization": f"Bearer {OPENROUTER_API_KEY}",
            "HTTP-Referer": "https://fyp-teacher.local",  # Adjust for OpenRouter compliance
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
            raise Exception(f"❌ Phi API Error {response.status_code}: {response.text}")

    except Exception as e:
        print("❌ API Call Failed:", e)
        return None

# === Prompt Formatter ===
def generate_lecture_plan_prompt(chapter, topic, clos, bloom, resources):
    resource_str = "\n".join([f"- {r['title']}: {r.get('description', '')}" for r in resources])
    clo_str = "\n".join([f"- {c}" for c in clos])

    return f"""
You are a Computer Science teacher preparing a lecture plan.

**Chapter:** {chapter}  
**Topic:** {topic}  
**CLOs:**  
{clo_str}  
**Bloom’s Level:** {bloom}  
**Available Resources:**  
{resource_str}  

Generate a 4-step lecture plan (45 minutes total). Follow this format:
1. **Introduction:** Hook or engaging start  
2. **Explanation:** Key concepts with visuals or activities  
3. **Activity:** Student engagement task or mini-quiz  
4. **Wrap-up:** Summary + common errors to clarify  

Use clear markdown with emojis, and keep it practical for teachers.
"""

# === Main UI Page ===
def lecture_planner_page(tab_key):
    key_prefix = f"lecture_{tab_key}"
    st.title("🧑‍🏫 AI-Powered Lecture Planner")

    curriculum = get_curriculum_data()
    if not curriculum or "chapters" not in curriculum:
        st.error("No curriculum data found.")
        return

    chapters = curriculum["chapters"]
    selected_chapter = st.selectbox("📘 Select Chapter", list(chapters.keys()), key=f"{key_prefix}_chapter")

    topics = chapters[selected_chapter].get("topics", {})
    selected_topic = st.selectbox("🔍 Select Topic", list(topics.keys()), key=f"{key_prefix}_topic")

    topic_data = topics[selected_topic]
    bloom = topic_data.get("bloom", "Understand")
    topic_clos = topic_data.get("clos", [])

    st.markdown(f"📌 **Bloom’s Level:** {bloom}")
    st.markdown("🎯 **Associated Topical Learning Objectives:**")
    for clo in topic_clos:
        st.markdown(f"- {clo}")

    resources = get_resources_by_topic(selected_topic) or []

    st.subheader("📚 Available Resources")
    for i, r in enumerate(resources[:5]):
        with st.expander(f"{i+1}. {r['title']}", expanded=False):
            st.markdown(f"**Type:** {r.get('type', 'N/A')}")
            st.markdown(f"**Description:** {r.get('description', 'No description')}") 
            if r.get("url"):
                st.markdown(f"[🔗 Open Resource]({r['url']})")

    if st.button("✨ Generate AI Lecture Plan", type="primary", key=f"{key_prefix}_generate"):
        with st.spinner("Generating lecture plan..."):
            prompt = generate_lecture_plan_prompt(
                selected_chapter,
                selected_topic,
                topic_clos,
                bloom,
                resources
            )

            response = call_phi4(
                prompt,
                system_msg="You are a curriculum expert and teacher assistant generating practical lecture plans.",
                max_tokens=1200
            )

            if response:
                st.subheader("📖 Generated Lecture Plan")
                st.markdown(response)
            else:
                st.error("⚠️ Failed to generate lecture plan. Please try again later.")
