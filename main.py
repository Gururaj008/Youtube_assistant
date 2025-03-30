import streamlit as st
import textwrap
import os
from youtube_transcript_api import YouTubeTranscriptApi
import faiss
import numpy as np
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
import google.generativeai as palm  # Import the package as a whole

# Set the Gemini API key from st.secrets
GEMINI_API_KEY = st.secrets["gemini_api_key"]
palm.configure(api_key=GEMINI_API_KEY)

def get_transcript(video_url):
    # Extract the YouTube video ID
    video_id = video_url.split("v=")[-1].split("&")[0]
    try:
        transcript = YouTubeTranscriptApi.get_transcript(video_id)
        return " ".join([entry["text"] for entry in transcript])
    except Exception as e:
        st.error(f"Error retrieving transcript: {e}")
        return None

def vdb_from_url(url):
    transcript = get_transcript(url)
    if not transcript:
        return None

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = text_splitter.split_text(transcript)
    
    embedding_model = OpenAIEmbeddings()
    embeddings = embedding_model.embed_documents(chunks)
    
    index = faiss.IndexFlatL2(len(embeddings[0]))
    index.add(np.array(embeddings))
    
    return {"index": index, "chunks": chunks, "embedding_model": embedding_model}

def find_relevant_chunk(db, query):
    embedding = db["embedding_model"].embed_documents([query])[0]
    distances, indices = db["index"].search(np.array([embedding]), k=1)
    return db["chunks"][indices[0][0]] if distances[0][0] < 0.5 else None

def response_for_query(db, query):
    relevant_chunk = find_relevant_chunk(db, query)
    
    if not relevant_chunk:
        return "Sorry! The question is out of the current context."
    
    prompt = f"Based on the following transcript excerpt, answer the question:\n\n{relevant_chunk}\n\nQuestion: {query}"
    response = palm.generate_text(prompt=prompt, model="gemini-pro")
    # Adjust based on the response object's structure (commonly, response.result contains the answer)
    return response.result if hasattr(response, "result") else "Sorry! Unable to generate an answer."

# Streamlit UI
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Agdasima');
    .custom-text-01 { font-family: 'Agdasima', sans-serif; font-size: 60px; color: cyan; }
    </style>
    <p class="custom-text-01">YouTube Assistant using LangChain and Gemini</p>
    """, unsafe_allow_html=True)

st.divider()
st.markdown("""
    <div style="text-align: justify">
        This Streamlit application extracts relevant answers from YouTube videos using LangChain and Google Gemini.
        Enter a YouTube URL and your query, and the system will search the video's transcript for the most relevant
        information and generate a response using Gemini.
    </div>
    """, unsafe_allow_html=True)

st.write("")
url = st.text_input("YouTube URL", "Please enter here..", key=1)
st.write("")
que = st.text_input("Your Question", "Please enter here..", key=2)
st.write("")
st.write("")

if st.button("Fetch the answer..", use_container_width=True):
    db = vdb_from_url(url)
    if not db:
        st.error("Could not retrieve transcript. Please check the URL.")
    else:
        response = response_for_query(db, que)
        wrapper = textwrap.TextWrapper(width=100)
        wrapped_text = wrapper.fill(response)
        st.write("")
        st.write("")
        st.write(wrapped_text)
