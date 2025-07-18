import streamlit as st
import firebase_admin
from firebase_admin import credentials, firestore

if not firebase_admin._apps:
    cred = credentials.Certificate({
        "type": st.secrets["FIREBASE_TYPE"],
        "project_id": st.secrets["FIREBASE_PROJECT_ID"],
        "private_key_id": st.secrets["FIREBASE_PRIVATE_KEY_ID"],
        "private_key": st.secrets["FIREBASE_PRIVATE_KEY"].replace("\\n", "\n"),
        "client_email": st.secrets["FIREBASE_CLIENT_EMAIL"],
        "client_id": st.secrets["FIREBASE_CLIENT_ID"],
        "auth_uri": st.secrets["FIREBASE_AUTH_URI"],
        "token_uri": st.secrets["FIREBASE_TOKEN_URI"],
        "auth_provider_x509_cert_url": st.secrets["FIREBASE_AUTH_PROVIDER_X509_CERT_URL"],
        "client_x509_cert_url": st.secrets["FIREBASE_CLIENT_X509_CERT_URL"]
    })
    firebase_admin.initialize_app(cred)

db = firestore.client()

# 📦 Modular Ref Helpers
def get_user_ref(email):
    return db.collection("students").document(email)

def get_progress_ref(email):
    return db.collection("progress").document(email)

def update_overall_stats(email):
    progress_ref = get_progress_ref(email)
    topics = progress_ref.collection("topics").stream()

    total = 0
    mastered = 0
    for doc in topics:
        data = doc.to_dict()
        total += 1
        if data.get("status") == "Mastered":
            mastered += 1

    stats = {
        "total_topics": total,
        "mastered_topics": mastered,
        "last_updated": firestore.SERVER_TIMESTAMP
    }
    progress_ref.set({"overall_stats": stats}, merge=True)
