import streamlit as st
import textwrap
# import os # Not used
from youtube_transcript_api import YouTubeTranscriptApi
import faiss
import numpy as np
# from langchain.embeddings import OpenAIEmbeddings # Removed OpenAI
from langchain_google_genai import GoogleGenerativeAIEmbeddings # Use Google Embeddings
# from langchain.vectorstores import FAISS # Using faiss directly
from langchain.text_splitter import RecursiveCharacterTextSplitter
import google.generativeai as genai # Use standard alias 'genai'

# --- Configuration ---
# Set the Gemini API key from st.secrets
# Ensure you have a secret named 'GOOGLE_API_KEY' in Streamlit Cloud
if "GOOGLE_API_KEY" not in st.secrets:
    st.error("Google API Key not found. Please add it as 'GOOGLE_API_KEY' to your Streamlit secrets.")
    st.stop()

GEMINI_API_KEY = st.secrets["GOOGLE_API_KEY"]
genai.configure(api_key=GEMINI_API_KEY)
# --- End Configuration ---

def get_transcript(video_url):
    """Extracts transcript text from a YouTube video URL."""
    try:
        # Extract YouTube video ID robustly
        video_id = None
        if "v=" in video_url:
            video_id = video_url.split("v=")[1].split("&")[0]
        elif "youtu.be/" in video_url:
            video_id = video_url.split("youtu.be/")[1].split("?")[0]

        if not video_id:
            st.error("Could not extract YouTube Video ID from the URL.")
            return None

        transcript_list = YouTubeTranscriptApi.get_transcript(video_id)
        transcript_text = " ".join([entry["text"] for entry in transcript_list])
        return transcript_text
    except Exception as e:
        st.error(f"Error retrieving transcript for video ID '{video_id}': {e}")
        return None

def create_vector_index(text):
    """Creates a FAISS index from text chunks using Google Embeddings."""
    if not text:
        return None

    try:
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100) # Increased chunk size slightly
        chunks = text_splitter.split_text(text)

        if not chunks:
            st.warning("Text could not be split into chunks.")
            return None

        # Use Google Embeddings via Langchain integration
        embedding_model = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        embeddings = embedding_model.embed_documents(chunks)

        # Ensure embeddings are valid
        if not embeddings or not embeddings[0]:
             st.error("Failed to generate embeddings for the text chunks.")
             return None

        dimension = len(embeddings[0])
        index = faiss.IndexFlatL2(dimension) # Use L2 distance
        index.add(np.array(embeddings).astype('float32')) # Add embeddings as float32 numpy array

        return {"index": index, "chunks": chunks, "embedding_model": embedding_model}
    except Exception as e:
        st.error(f"Error creating vector index: {e}")
        return None

def find_relevant_chunks(vector_index, query, k=3):
    """Finds relevant chunks in the index based on the query."""
    try:
        # Embed the query using the same model
        query_embedding = vector_index["embedding_model"].embed_query(query)

        # Search the index
        distances, indices = vector_index["index"].search(np.array([query_embedding]).astype('float32'), k=k)

        # Return the text of the relevant chunks
        relevant_docs = [vector_index["chunks"][i] for i in indices[0]]
        return "\n\n".join(relevant_docs) # Join chunks with newlines for context
    except Exception as e:
        st.error(f"Error searching vector index: {e}")
        return None

      
def generate_response_with_gemini(context, query):
    """Generates an answer using Gemini based on context and query."""
    try:
        # Use the current standard API for Gemini
        model = genai.GenerativeModel('gemini-2.0-flash-thinking-exp-01-21')

        prompt = f"""You are a helpful YouTube assistant.
Answer the following question based *only* on the provided transcript excerpt(s).
If the information is not in the excerpt(s), say "Sorry! The question is out of the current context".
Do not mention the words "transcript" or "excerpt" in your response.

Transcript Excerpt(s):
{context}

Question: {query}

Answer:"""

        # Generate content
        response = model.generate_content(prompt)

        # Safely access the response text
        if response.parts:
            return response.text
        elif response.prompt_feedback.block_reason:
             # Handle cases where content was blocked due to safety settings
             return f"Sorry, the response was blocked due to: {response.prompt_feedback.block_reason.name}"
        else:
             # Handle other cases where no text part is available
             return "Sorry! Gemini could not generate an answer for this query based on the context."

    except Exception as e:
        st.error(f"Error generating response with Gemini: {e}")
        return "Sorry! An error occurred while generating the answer."

def response_for_query(db, query): # Keep the response_for_query function as a wrapper
    relevant_context = find_relevant_chunks(db, query)
    if not relevant_context:
        return "Sorry! The question is out of the current context."
    return generate_response_with_gemini(relevant_context, query) # Call the new function

    

# --- Streamlit UI ---
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Agdasima');
    .custom-text-01 { font-family: 'Agdasima', sans-serif; font-size: 60px; color: cyan; }
    </style>
    <p class="custom-text-01">YouTube Assistant using Google Gemini</p>
    """, unsafe_allow_html=True)

st.divider()
st.markdown("""
    <div style="text-align: justify">
        This Streamlit application extracts relevant answers from YouTube videos using Google Gemini.
        Enter a YouTube URL and your query. The system retrieves the video's transcript, creates vector embeddings using Google's models,
        finds the most relevant parts of the transcript based on your question, and then uses the Gemini model to generate a response based on that context.
    </div>
    """, unsafe_allow_html=True)

st.write("")
url = st.text_input("YouTube URL", "", key="url_input", placeholder="e.g., https://www.youtube.com/watch?v=...") # Added placeholder
st.write("")
que = st.text_input("Your Question", "", key="question_input", placeholder="e.g., What is the main topic?") # Added placeholder
st.write("")
st.write("")

if st.button("Fetch the answer..", use_container_width=True):
    if not url:
        st.warning("Please enter a YouTube URL.")
    elif not que:
        st.warning("Please enter your question.")
    else:
        vector_index = None # Initialize vector_index
        with st.spinner("Processing YouTube video... (This may take a moment)"):
            transcript_text = get_transcript(url)
            if transcript_text:
                vector_index = create_vector_index(transcript_text)

        if vector_index:
            with st.spinner("Finding relevant information and generating answer..."):
                relevant_context = find_relevant_chunks(vector_index, que)
                if relevant_context:
                    answer = generate_response_with_gemini(relevant_context, que)
                else:
                    answer = "Could not find relevant information in the transcript for your question."

                wrapper = textwrap.TextWrapper(width=100) # Adjust width as needed
                wrapped_text = wrapper.fill(answer)
                st.write("")
                st.write("### Answer:")
                st.markdown(wrapped_text) # Use markdown for better formatting
       
