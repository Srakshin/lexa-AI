import os
import streamlit as st
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI


os.environ["GOOGLE_API_KEY"] = "AIzaSyDydWxM_3IoML4ZPSe-YAlBQOZvXGCz8PI"


PDF_FILES = [
    "April25.pdf",
    "CCApril25.pdf",
    "CCFebruary25.pdf",
    "CCJanuary25.pdf",
    "CCMarch26.pdf",
    "CCMay25.pdf",
    "February25.pdf",
    "January25.pdf",
    "March25.pdf"
]
INDEX_DIR = "faiss_clat_index"
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50
EMBED_MODEL = "all-MiniLM-L6-v2"


st.set_page_config(page_title="CLAT Chatbot", layout="wide")

st.markdown("""
    <style>
    body {
        background-color: #1e1f23;
        color: #f1f1f1;
    }
    .main {
        background-color: #1e1f23;
    }
    .chat-container {
    width: 100%;
    padding: 1rem;
}

    .user-msg, .bot-msg {
    padding: 1rem;
    border-radius: 1rem;
    margin: 0.75rem 0;
    width: 100%; /* ✅ full width */
    word-wrap: break-word;
    white-space: pre-wrap; /* ✅ preserves line breaks */
    background-color: #2e2f34;
}

    .user-msg {
        background-color: #5c5f66;
        color: #fff;
        margin-left: auto;
        text-align: left;
        display: flex;
        align-items: center;
        gap: 0.6rem;
    }
    .bot-msg {
        background-color: #2e2f34;
        color: #f8f9fa;
        margin-right: auto;
        text-align: left;
        display: flex;
        align-items: center;
        gap: 0.6rem;
    }
    .emoji-bubble {
        font-size: 1.3rem;
    }
    .title {
        text-align: center;
        font-size: 2rem;
        font-weight: bold;
        margin-top: 1rem;
        color: #e0e0e0;
    }
    .caption {
        text-align: center;
        font-size: 1rem;
        color: #aaa;
        margin-bottom: 2rem;
    }
    input[type="text"] {
        background-color: #2a2b2f !important;
        color: white !important;
    }
    </style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_documents():
    docs = []
    for file in PDF_FILES:
        loader = PyPDFLoader(file)
        pages = loader.load()
        splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
        docs.extend(splitter.split_documents(pages))
    return docs

# ---------- Vectorstore ----------
@st.cache_resource
def get_vectorstore(_docs):
    if os.path.exists(INDEX_DIR):
        return FAISS.load_local(
            INDEX_DIR,
            HuggingFaceEmbeddings(model_name=EMBED_MODEL),
            allow_dangerous_deserialization=True
        )
    embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL)
    db = FAISS.from_documents(_docs, embeddings)
    db.save_local(INDEX_DIR)
    return db

@st.cache_resource
def build_qa_chain():
    docs = load_documents()
    db = get_vectorstore(docs)
    retriever = db.as_retriever(search_kwargs={"k": 10})

    custom_prompt = PromptTemplate.from_template("""
You are a helpful and strict CLAT-AILET study assistant. Your job is to answer legal, constitutional, current affairs, or general knowledge questions **strictly** based on the context provided from the compendium.

When answering, follow this structure:

1. Start with a **clear factual introductory sentence** answering the question.
2. Add 2–4 bullet points (•) for key facts or syllabus-relevant details.
3. End with a CLAT exam prep tip, such as: “Refer to trusted sources like The Hindu or PIB for latest updates.”

⚠️ If the question is unrelated to CLAT or not found in context, politely say: 
“Sorry, I can only assist with CLAT or AILET related questions based on your syllabus or compendium content.”

Question: {question}
Context: {context}
Answer:
""")


    llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.3)

    return RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        chain_type_kwargs={"prompt": custom_prompt}
    )


if "chat" not in st.session_state:
    st.session_state.chat = []

qa_chain = build_qa_chain()
def get_custom_static_answer(question):
    q = question.lower()
    if "syllabus" in q:
        return """**CLAT 2026 Syllabus**

The CLAT UG exam covers 5 key areas:

- **English Language**: Reading comprehension, grammar, vocabulary.
- **Current Affairs & GK**: National/international events, legal updates.
- **Legal Reasoning**: Legal principles, case laws, application-based Qs.
- **Logical Reasoning**: Critical reasoning, puzzles, assumptions.
- **Quantitative Techniques**: Basic math (up to class 10).

Refer to official guides like www.consortiumofnlus.ac.in for updates."""

    elif "weightage" in q:
        return """**CLAT Section-Wise Weightage (2026 pattern)**

- English: ~20%
- GK & Current Affairs: ~25%
- Legal Reasoning: ~25%
- Logical Reasoning: ~20–25%
- Quantitative Techniques: ~10–12%

Each section is comprehension-based and tests reasoning as well as knowledge."""

    elif "6 month" in q or "study plan" in q or "how should i plan" in q:
        return """**6-Month CLAT Study Plan**

- **Months 1–2**: Focus on concepts, daily GK, basic logic, vocabulary.
- **Months 3–4**: Start mocks weekly, revise weak topics.
- **Months 5–6**: Take full mocks, revise flashcards, track speed & accuracy.

Use books like R.S. Aggarwal (English & Reasoning), Universal’s CLAT Guide (Legal)."""

    elif "hindu" in q and "enough" in q:
        return """**Is The Hindu enough for CLAT GK?**

The Hindu is great, but also use:

- **PIB**, **GKToday**, and **AffairsCloud** for quick updates
- Monthly compilations for revision
- Make short notes to track legal + national updates.

Use The Hindu for editorial reading + in-depth analysis."""

    return None



with st.sidebar:
    st.markdown("## 🧠 Lexa")
    st.markdown("Talk to your Powerful Lexa AI")


st.markdown("<div class='chat-container'>", unsafe_allow_html=True)
st.markdown("<div class='title'>💬 Lexa AI Assistant</div>", unsafe_allow_html=True)
st.markdown("<div class='caption'>Ask anything from your CLAT Compendiums — I’ve got your back!</div>", unsafe_allow_html=True)

for sender, msg in st.session_state.chat:
    if sender == "user":
        st.markdown(f"**🧑‍💻 You:**\n\n{msg}", unsafe_allow_html=False)
    else:
        st.markdown(f"**🤖 CLATBot:**\n\n{msg}", unsafe_allow_html=False)



st.markdown("</div>", unsafe_allow_html=True)


with st.form("chat_form", clear_on_submit=True):
    user_input = st.text_input("💬 Your question:", placeholder="e.g. Who was appointed as CEC in April 2025?")
    submit_btn = st.form_submit_button("Send")

if submit_btn and user_input:
    st.session_state.chat.append(("user", user_input))

  
    vague_inputs = ["ok", "okay", "hmm", "cool", "nice", "great", "k", "alright"]
    greetings = ["hi", "hello", "hey", "yo", "hola"]

    off_topic_keywords = [
        "movie", "game", "cricket", "football", "actor", "celebrity", "instagram",
        "entertainment", "music", "anime", "netflix", "dance", "song", "meme", "bollywood"
    ]
    study_keywords = [
        "study", "plan", "strategy", "revise", "prepare", "motivate", "revision",
        "schedule", "learn", "how to", "tips", "manage", "concentrate"
    ]

    question_lower = user_input.strip().lower()

    with st.spinner("🤖 Thinking..."):
        if question_lower in vague_inputs:
            response = "Please ask a specific question related to CLAT, law, or your study plan."

        elif question_lower in greetings:
            response = "👋 Hey there! How can I help you with your CLAT or AILET prep today?"


        elif any(word in question_lower for word in study_keywords):
            
            response = ChatGoogleGenerativeAI(model="gemini-1.5-flash").invoke(
    f"You are a strict but supportive mentor helping a student prepare for CLAT. Be concise and helpful. Question: {user_input}"
).content


        else:
    
            response = get_custom_static_answer(user_input)

            if not response:
     
                response = qa_chain.run(user_input)

            if not response.strip() or "i don't know" in response.lower() or "cannot find" in response.lower():
                response = ChatGoogleGenerativeAI(model="gemini-1.5-flash").invoke(
                    f"You are a CLAT and law preparation expert. Answer this question concisely: {user_input}"
                ).content

   
            if not response.strip() or "i don't know" in response.lower() or "cannot find" in response.lower():
                if "syllabus" in question_lower or "practice" in question_lower or "questions" in question_lower:
                    response = """
**CLAT 2026 Syllabus Overview**

The CLAT exam tests 5 major sections:

1. **English Language** – Reading comprehension, grammar, vocabulary, and inference.  
2. **Current Affairs & GK** – Legal & national current events, static GK.  
3. **Legal Reasoning** – Legal principles, case laws, and application-based questions.  
4. **Logical Reasoning** – Puzzles, arguments, critical reasoning.  
5. **Quantitative Techniques** – Class 10-level math: percentages, profit-loss, data interpretation.  

---

**Sample Practice Questions:**

**English:**  
_What does the word "transcend" mean in the context of growth?_  
A) Destroy B) Rise above C) Delay D) Avoid  
✅ Answer: **B**

**Legal Reasoning:**  
_If a law says "No citizen may protest past 10 PM" and Alex protests at 11 PM, what principle applies?_  
✅ Answer: **Violation of statutory law**

**Logical Reasoning:**  
_All poets are dreamers. Some dreamers are realists._  
Conclude:  
A) Some poets are realists  
B) All poets are realists  
C) Some dreamers are not poets  
D) No valid conclusion  
✅ Answer: **D**

---

*Tip:* Read editorials daily. Practice with mocks. Revise legal maxims & recent appointments weekly.
"""

            else:
                # Let Gemini answer more freely for general law/CLAT prep queries
                response = ChatGoogleGenerativeAI(model="gemini-1.5-flash").invoke(
    f"You are a strict but supportive mentor helping a student prepare for CLAT. Be concise and helpful. Question: {user_input}"
).content



    st.session_state.chat.append(("bot", response))
    st.rerun()



st.markdown("</div>", unsafe_allow_html=True)
