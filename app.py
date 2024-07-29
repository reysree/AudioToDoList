import streamlit as st
from openai import OpenAI
from dotenv import load_dotenv
import os
from groq import Groq

# Load environment variables
load_dotenv()

# Initialize OpenAI client
openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Initialize Groq client
groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))

def transcribe_audio(file):
    transcription = openai_client.audio.transcriptions.create(
        model="whisper-1", 
        file=file
    )
    return transcription.text

def generate_todo_list(transcription_text):
    chat_completion = groq_client.chat.completions.create(
        messages=[
            {
                "role": "system",
                "content": """You are an efficient personal assistant tasked with creating a clear, concise, and actionable to-do list based on the user's voice transcription. Follow these guidelines:

1. Extract only specific, actionable tasks from the transcription.
2. Omit any introductory or explanatory text (e.g., "Here is a list of actionable tasks").
3. Don't include section headers or titles (e.g., "Today's To-Do List").
4. Start each task with a verb and be specific.
5. If time or location is mentioned for a task, include it.
6. Limit the list to a maximum of 10 items.
7. Prioritize tasks if possible, listing more urgent or important tasks first.
8. If the transcription includes vague ideas, convert them into concrete actions.

Example of a good to-do list:
1. Call Dr. Smith at 2 PM to schedule annual check-up
2. Buy groceries: milk, eggs, bread, and vegetables
3. Complete project report draft by 5 PM
4. Email team about tomorrow's 10 AM meeting
5. Go for a 30-minute walk in the park
6. Pay electricity bill online
7. Prepare presentation slides for client meeting
8. Read chapter 3 of "Project Management Basics"
"""
            },
            {
                "role": "user",
                "content": transcription_text,
            }
        ],
        model="llama-3.1-8b-instant",
    )
    return chat_completion.choices[0].message.content

st.title("To-Do List Generator from Audio")

# Initialize session state for todo list and its state
if 'todo_items' not in st.session_state:
    st.session_state.todo_items = []
if 'todo_state' not in st.session_state:
    st.session_state.todo_state = []

uploaded_file = st.file_uploader("Upload an audio file", type=["mp3", "wav"])

if uploaded_file is not None and 'transcription_done' not in st.session_state:
    with st.spinner("Transcribing audio..."):
        transcription_text = transcribe_audio(uploaded_file)
    
    with st.spinner("Generating to-do list..."):
        todo_list = generate_todo_list(transcription_text)
    
    st.session_state.todo_items = todo_list.split('\n')
    st.session_state.todo_state = [False] * len(st.session_state.todo_items)
    st.session_state.transcription_done = True

if st.session_state.todo_items:
    st.write("To-Do List:")
    for i, item in enumerate(st.session_state.todo_items):
        key = f"todo_{i}"
        st.session_state.todo_state[i] = st.checkbox(item, key=key, value=st.session_state.todo_state[i])
    
    # st.write("To-Do List State:")
    # for item, state in zip(st.session_state.todo_items, st.session_state.todo_state):
    #     st.write(f"{item}: {'Done' if state else 'Not Done'}")

# Add a button to clear the to-do list and reset the state
if st.button("Clear To-Do List"):
    st.session_state.todo_items = []
    st.session_state.todo_state = []
    st.session_state.pop('transcription_done', None)
    st.rerun()