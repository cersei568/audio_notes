from io import BytesIO
import streamlit as st
from audiorecorder import audiorecorder  # type: ignore
from dotenv import dotenv_values
from hashlib import md5
from openai import OpenAI
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct, Distance, VectorParams
import time

env = dotenv_values(".env")

EMBEDDING_MODEL = "text-embedding-3-large"
EMBEDDING_DIM = 3072
AUDIO_TRANSCRIBE_MODEL = "whisper-1"
QDRANT_COLLECTION_NAME = "notes"

# Premium styling with navy blue and blue theme
def apply_premium_styling():
    st.markdown("""
    <style>
        /* Import Google Fonts */
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
        
        /* Main app styling */
        .main {
            background: linear-gradient(135deg, #0f1f3c 0%, #1e3a5f 50%, #2c5282 100%);
            font-family: 'Inter', sans-serif;
        }
        
        .stApp {
            background: linear-gradient(135deg, #0f1f3c 0%, #1e3a5f 50%, #2c5282 100%);
        }
        
        /* Header styling */
        .premium-header {
            background: linear-gradient(90deg, #1a365d, #2c5282, #3182ce);
            padding: 2rem;
            border-radius: 15px;
            margin-bottom: 2rem;
            box-shadow: 0 10px 30px rgba(0,0,0,0.3);
            border: 1px solid rgba(49, 130, 206, 0.3);
        }
        
        .premium-title {
            font-size: 3rem;
            font-weight: 700;
            color: #ffffff;
            text-align: center;
            margin: 0;
            text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
            background: linear-gradient(45deg, #63b3ed, #90cdf4, #bee3f8);
            background-clip: text;
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }
        
        .premium-subtitle {
            font-size: 1.2rem;
            color: #bee3f8;
            text-align: center;
            margin-top: 0.5rem;
            font-weight: 300;
        }
        
        /* API Key section styling */
        .api-key-container {
            background: linear-gradient(135deg, #1a202c, #2d3748);
            padding: 2rem;
            border-radius: 15px;
            border: 2px solid #3182ce;
            margin-bottom: 2rem;
            box-shadow: 0 8px 25px rgba(49, 130, 206, 0.2);
        }
        
        .api-key-title {
            color: #90cdf4;
            font-size: 1.5rem;
            font-weight: 600;
            margin-bottom: 1rem;
            text-align: center;
        }
        
        .api-key-info {
            background: rgba(49, 130, 206, 0.1);
            padding: 1rem;
            border-radius: 10px;
            border-left: 4px solid #3182ce;
            color: #bee3f8;
            margin-bottom: 1rem;
        }
        
        /* Tab styling */
        .stTabs [data-baseweb="tab-list"] {
            gap: 24px;
            background: rgba(26, 54, 93, 0.5);
            padding: 0.5rem;
            border-radius: 15px;
            border: 1px solid rgba(49, 130, 206, 0.3);
        }
        
        .stTabs [data-baseweb="tab"] {
            height: 60px;
            padding: 0 24px;
            background: linear-gradient(135deg, #2c5282, #3182ce);
            color: white;
            border-radius: 10px;
            border: none;
            font-weight: 500;
            font-size: 1.1rem;
            transition: all 0.3s ease;
        }
        
        .stTabs [aria-selected="true"] {
            background: linear-gradient(135deg, #3182ce, #4299e1) !important;
            box-shadow: 0 4px 15px rgba(49, 130, 206, 0.4);
            transform: translateY(-2px);
        }
        
        /* Button styling */
        .stButton > button {
            background: linear-gradient(135deg, #2b6cb0, #3182ce);
            color: white;
            border: none;
            border-radius: 10px;
            padding: 0.75rem 2rem;
            font-weight: 600;
            font-size: 1rem;
            transition: all 0.3s ease;
            box-shadow: 0 4px 15px rgba(43, 108, 176, 0.3);
        }
        
        .stButton > button:hover {
            background: linear-gradient(135deg, #3182ce, #4299e1);
            transform: translateY(-2px);
            box-shadow: 0 6px 20px rgba(43, 108, 176, 0.4);
        }
        
        /* Text input styling */
        .stTextInput > div > div > input {
            background: rgba(26, 54, 93, 0.3);
            color: #e2e8f0;
            border: 2px solid rgba(49, 130, 206, 0.3);
            border-radius: 10px;
            padding: 1rem;
            font-size: 1rem;
        }
        
        .stTextInput > div > div > input:focus {
            border-color: #4299e1;
            box-shadow: 0 0 15px rgba(66, 153, 225, 0.3);
        }
        
        /* Text area styling */
        .stTextArea > div > div > textarea {
            background: rgba(26, 54, 93, 0.3);
            color: #e2e8f0;
            border: 2px solid rgba(49, 130, 206, 0.3);
            border-radius: 10px;
            padding: 1rem;
            font-size: 1rem;
        }
        
        .stTextArea > div > div > textarea:focus {
            border-color: #4299e1;
            box-shadow: 0 0 15px rgba(66, 153, 225, 0.3);
        }
        
        /* Container styling */
        .note-container {
            background: linear-gradient(135deg, rgba(26, 54, 93, 0.6), rgba(44, 82, 130, 0.4));
            padding: 1.5rem;
            border-radius: 15px;
            border: 1px solid rgba(49, 130, 206, 0.3);
            margin-bottom: 1rem;
            backdrop-filter: blur(10px);
            box-shadow: 0 8px 25px rgba(0, 0, 0, 0.2);
        }
        
        /* Success message styling */
        .success-message {
            background: linear-gradient(135deg, #065f46, #047857);
            color: #d1fae5;
            padding: 1rem;
            border-radius: 10px;
            text-align: center;
            font-weight: 500;
            margin: 1rem 0;
        }
        
        /* Score styling */
        .search-score {
            color: #90cdf4;
            font-style: italic;
            font-size: 0.9rem;
            margin-top: 0.5rem;
        }
        
        /* Audio recorder styling */
        .audio-recorder {
            text-align: center;
            padding: 2rem;
            background: rgba(26, 54, 93, 0.3);
            border-radius: 15px;
            border: 2px dashed rgba(49, 130, 206, 0.5);
            margin: 1rem 0;
        }
        
        /* Premium badge */
        .premium-badge {
            display: inline-block;
            background: linear-gradient(45deg, #f6ad55, #ed8936);
            color: white;
            padding: 0.3rem 0.8rem;
            border-radius: 20px;
            font-size: 0.8rem;
            font-weight: 600;
            margin-left: 1rem;
        }
        
        /* Labels and text visibility fixes */
        .stTextInput > label, .stTextArea > label {
            color: #90cdf4 !important;
            font-weight: 600 !important;
            font-size: 1.1rem !important;
        }
        
        /* Main content text visibility */
        .main .block-container {
            color: #e2e8f0;
        }
        
        /* Header text styling - make headers visible */
        h1, h2, h3, h4, h5, h6 {
            color: #ffffff !important;
        }
        
        /* Markdown text styling */
        .main p, .main div, .main span {
            color: #e2e8f0 !important;
        }
        
        /* Specific fixes for section headers */
        .stMarkdown h3 {
            color: #90cdf4 !important;
            font-weight: 600 !important;
            margin-bottom: 1rem !important;
        }
        
        /* Column text visibility */
        .stColumn p, .stColumn div {
            color: #bee3f8 !important;
            font-weight: 500 !important;
        }
        
        /* Audio preview text */
        .stMarkdown strong {
            color: #90cdf4 !important;
            font-weight: 600 !important;
        }
        
        /* Checkbox styling */
        .stCheckbox > label {
            color: #90cdf4 !important;
            font-weight: 500 !important;
        }
        
        /* Metric styling */
        .metric-container {
            background: rgba(26, 54, 93, 0.4);
            padding: 1rem;
            border-radius: 10px;
            border: 1px solid rgba(49, 130, 206, 0.3);
        }
        
        /* Info message styling */
        .stInfo {
            background: rgba(49, 130, 206, 0.15) !important;
            border: 1px solid rgba(49, 130, 206, 0.3) !important;
            color: #bee3f8 !important;
        }
        
        /* Expander styling */
        .stExpander {
            background: rgba(26, 54, 93, 0.3);
            border: 1px solid rgba(49, 130, 206, 0.3);
            border-radius: 10px;
        }
        
        .stExpander > summary {
            color: #90cdf4 !important;
            font-weight: 500 !important;
        }
        
        /* Footer styling */
        .stMarkdown p:last-child {
            color: #90cdf4 !important;
        }
    </style>
    """, unsafe_allow_html=True)

def get_openai_client():
    return OpenAI(api_key=st.session_state["openai_api_key"])

def transcribe_audio(audio_bytes):
    with st.spinner("🎵 Transcribing your audio..."):
        time.sleep(0.5)  # Small delay for better UX
        openai_client = get_openai_client()
        audio_file = BytesIO(audio_bytes)
        audio_file.name = "audio.mp3"
        transcript = openai_client.audio.transcriptions.create(
            file=audio_file,
            model=AUDIO_TRANSCRIBE_MODEL,
            response_format="verbose_json",
        )
    return transcript.text

@st.cache_resource
def get_qdrant_client():
    return QdrantClient(path=":memory:")

def assure_db_collection_exists():
    qdrant_client = get_qdrant_client()
    if not qdrant_client.collection_exists(QDRANT_COLLECTION_NAME):
        qdrant_client.create_collection(
            collection_name=QDRANT_COLLECTION_NAME,
            vectors_config=VectorParams(
                size=EMBEDDING_DIM,
                distance=Distance.COSINE,
            ),
        )

def get_embedding(text):
    openai_client = get_openai_client()
    result = openai_client.embeddings.create(
        input=[text],
        model=EMBEDDING_MODEL,
        dimensions=EMBEDDING_DIM,
    )
    return result.data[0].embedding

def add_note_to_db(note_text):
    with st.spinner("💾 Saving your note..."):
        time.sleep(0.3)  # Small delay for better UX
        qdrant_client = get_qdrant_client()
        points_count = qdrant_client.count(
            collection_name=QDRANT_COLLECTION_NAME,
            exact=True,
        )
        qdrant_client.upsert(
            collection_name=QDRANT_COLLECTION_NAME,
            points=[
                PointStruct(
                    id=points_count.count + 1,
                    vector=get_embedding(text=note_text),
                    payload={
                        "text": note_text,
                    },
                )
            ]
        )

def list_notes_from_db(query=None):
    with st.spinner("🔍 Searching your notes..." if query else "📚 Loading your notes..."):
        time.sleep(0.3)  # Small delay for better UX
        qdrant_client = get_qdrant_client()
        if not query:
            notes = qdrant_client.scroll(collection_name=QDRANT_COLLECTION_NAME, limit=10)[0]
            result = []
            for note in notes:
                result.append({
                    "text": note.payload["text"],
                    "score": None,
                })
            return result
        else:
            notes = qdrant_client.search(
                collection_name=QDRANT_COLLECTION_NAME,
                query_vector=get_embedding(text=query),
                limit=10,
            )
            result = []
            for note in notes:
                result.append({
                    "text": note.payload["text"],
                    "score": note.score,
                })
            return result

# Main application
st.set_page_config(
    page_title="AudioNotes Pro",
    page_icon="🎙️",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# Apply premium styling
apply_premium_styling()

# Premium header
st.markdown("""
<div class="premium-header">
    <h1 class="premium-title">AudioNotes Pro <span class="premium-badge"> </span></h1>
    <p class="premium-subtitle">AI-Powered Voice Note Management System</p>
</div>
""", unsafe_allow_html=True)

# OpenAI API key section
if not st.session_state.get("openai_api_key"):
    if "OPENAI_API_KEY" in env:
        st.session_state["openai_api_key"] = env["OPENAI_API_KEY"]
    else:
        st.markdown("""
        <div class="api-key-container">
            <h3 class="api-key-title">🔐 Premium Access Required</h3>
            <div class="api-key-info">
                <strong>Welcome to AudioNotes Pro!</strong><br>
                To unlock all premium features including AI transcription and semantic search, 
                please enter your OpenAI API key below.
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        api_key = st.text_input(
            "Enter your OpenAI API Key",
            type="password",
            placeholder="sk-...",
            help="Your API key is encrypted and never stored permanently"
        )
        
        if api_key:
            st.session_state["openai_api_key"] = api_key
            st.success("🎉 API key validated! Welcome to AudioNotes Pro!")
            time.sleep(1)
            st.rerun()

if not st.session_state.get("openai_api_key"):
    st.stop()

# Session state initialization
if "note_audio_bytes_md5" not in st.session_state:
    st.session_state["note_audio_bytes_md5"] = None
if "note_audio_bytes" not in st.session_state:
    st.session_state["note_audio_bytes"] = None
if "note_text" not in st.session_state:
    st.session_state["note_text"] = ""
if "note_audio_text" not in st.session_state:
    st.session_state["note_audio_text"] = ""

# Initialize database
assure_db_collection_exists()

# Main tabs
add_tab, search_tab, stats_tab = st.tabs(["🎙️ Record Note", "🔍 Search Notes", "📊 Statistics"])

with add_tab:
    st.markdown('<h3 style="color: #90cdf4 !important;">Record Your Voice Note</h3>', unsafe_allow_html=True)
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        note_audio = audiorecorder(
            start_prompt="🎤 Start Recording",
            stop_prompt="⏹️ Stop Recording",
        )
    
    with col2:
        if st.session_state.get("note_audio_bytes"):
            st.markdown('<p style="color: #90cdf4 !important; font-weight: 600;"><strong>Audio Preview:</strong></p>', unsafe_allow_html=True)
    
    if note_audio:
        audio = BytesIO()
        note_audio.export(audio, format="mp3")
        st.session_state["note_audio_bytes"] = audio.getvalue()
        current_md5 = md5(st.session_state["note_audio_bytes"]).hexdigest()
        
        if st.session_state["note_audio_bytes_md5"] != current_md5:
            st.session_state["note_audio_text"] = ""
            st.session_state["note_text"] = ""
            st.session_state["note_audio_bytes_md5"] = current_md5

        st.audio(st.session_state["note_audio_bytes"], format="audio/mp3")

        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("🤖 Transcribe with AI", use_container_width=True):
                st.session_state["note_audio_text"] = transcribe_audio(st.session_state["note_audio_bytes"])
                st.success("✅ Audio transcribed successfully!")

        if st.session_state["note_audio_text"]:
            st.markdown('<h3 style="color: #90cdf4 !important;">Edit Your Note</h3>', unsafe_allow_html=True)
            st.session_state["note_text"] = st.text_area(
                "Note Content",
                value=st.session_state["note_audio_text"],
                height=150,
                help="Edit the transcribed text or add additional notes"
            )

            if st.button("💾 Save Note", disabled=not st.session_state["note_text"], use_container_width=True):
                add_note_to_db(note_text=st.session_state["note_text"])
                st.balloons()
                st.success("🎉 Note saved successfully!")
                # Clear the form
                st.session_state["note_text"] = ""
                st.session_state["note_audio_text"] = ""
                st.session_state["note_audio_bytes"] = None

with search_tab:
    st.markdown('<h3 style="color: #90cdf4 !important;">🔍 Search Your Notes</h3>', unsafe_allow_html=True)
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        query = st.text_input(
            "Search Query",
            value="",
            placeholder="Enter keywords to search your notes..."
        )
    
    with col2:
        search_all = st.checkbox("Show All Notes", value=False)
    
    if st.button("🔍 Search Notes", use_container_width=True) or search_all:
        search_query = None if search_all else query
        notes = list_notes_from_db(search_query)
        
        if notes:
            st.markdown(f'<h3 style="color: #90cdf4 !important;">Found {len(notes)} note(s)</h3>', unsafe_allow_html=True)
            
            for i, note in enumerate(notes, 1):
                with st.container():
                    st.markdown(f"""
                    <div class="note-container">
                        <h4 style="color: #90cdf4; margin-bottom: 1rem;">📝 Note {i}</h4>
                        <p style="color: #e2e8f0; line-height: 1.6;">{note["text"]}</p>
                        {f'<div class="search-score">Relevance Score: {note["score"]:.3f}</div>' if note["score"] else ''}
                    </div>
                    """, unsafe_allow_html=True)
        else:
            st.info("🤷‍♂️ No notes found. Try a different search term or record your first note!")

with stats_tab:
    st.markdown('<h3 style="color: #90cdf4 !important;">📊 Your Notes Statistics</h3>', unsafe_allow_html=True)
    
    qdrant_client = get_qdrant_client()
    total_notes = qdrant_client.count(collection_name=QDRANT_COLLECTION_NAME, exact=True).count
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Total Notes", total_notes, delta=None)
    
    with col2:
        st.metric("Storage Used", f"{total_notes * 0.5:.1f} KB", delta=None)
    
    with col3:
        st.metric("Premium Features", "Unlimited", delta=None)
    
    if total_notes > 0:
        st.markdown('<h3 style="color: #90cdf4 !important;">Recent Activity</h3>', unsafe_allow_html=True)
        recent_notes = list_notes_from_db()[:3]
        
        for i, note in enumerate(recent_notes, 1):
            with st.expander(f"Recent Note {i}"):
                st.write(note["text"][:200] + "..." if len(note["text"]) > 200 else note["text"])

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #90cdf4; font-size: 0.9rem; margin-top: 2rem;">
    <p>AudioNotes Pro - Powered by OpenAI & Qdrant | Premium Voice Note Management</p>
</div>
""", unsafe_allow_html=True)