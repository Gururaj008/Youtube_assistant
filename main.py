import traceback
from urllib.parse import urlparse, parse_qs

from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
import textwrap
import streamlit as st
import os
import google.generativeai as genai
from youtube_transcript_api import YouTubeTranscriptApi

# Set Google API Key from Streamlit secrets
if "GOOGLE_API_KEY" not in st.secrets:
    st.error("Google API Key not found. Please add it to your Streamlit secrets.")
    st.stop()

# Configure Google Generative AI with the key from secrets
genai.configure(api_key=st.secrets["GOOGLE_API_KEY"])

def get_video_id(url):
    """Extracts the video ID from a YouTube URL."""
    parsed_url = urlparse(url)
    query_params = parse_qs(parsed_url.query)
    video_id = query_params.get("v", [None])[0]
    if not video_id and parsed_url.hostname in ["youtu.be"]:
        # For shortened URLs like https://youtu.be/VIDEO_ID
        video_id = parsed_url.path.lstrip("/")
    return video_id

def load_youtube_transcript(url, proxies=None):
    """
    Loads the YouTube transcript using youtube_transcript_api.
    Pass proxies (as a dict) if provided.
    Returns a list containing a single Document with the full transcript.
    """
    video_id = get_video_id(url)
    if not video_id:
        raise ValueError("Invalid YouTube URL, unable to extract video ID.")
    transcript = YouTubeTranscriptApi.get_transcript(video_id, proxies=proxies)
    full_text = " ".join(segment["text"] for segment in transcript)
    return [Document(page_content=full_text, metadata={"video_id": video_id})]

def vdb_from_url(url):
    try:
        # Check if a proxy is provided in the secrets; expected as a dict
        proxies = st.secrets.get("PROXY", None)
        
        # Use the custom loader function to get the transcript as Document objects
        documents = load_youtube_transcript(url, proxies=proxies)

        if not documents:
            st.warning("Could not load transcript. Please check the URL and ensure the video has transcripts available.")
            return None

        # Split the transcript into manageable chunks
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        docs_splitted = text_splitter.split_documents(documents)

        if not docs_splitted:
            st.warning("Transcript loaded but could not be split into chunks.")
            return None

        # Create embeddings and build the FAISS vector index
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        db = FAISS.from_documents(docs_splitted, embeddings)
        return db

    except Exception as e:
        st.error(f"Error processing YouTube URL: {e}")
        print("Detailed traceback:", traceback.format_exc())
        return None

def response_for_query(db, query, k):
    # Perform similarity search on the FAISS index
    docs = db.similarity_search(query, k=k)
    page_content = " ".join([d.page_content for d in docs])

    # Initialize the LLM with deterministic output
    llm = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.0)

    # Define the prompt template
    prompt = PromptTemplate(
        input_variables=["question", "docs"],
        template="""
        You are a helpful YouTube assistant powered by Google AI.
        Your mission is to answer questions about videos based solely on the video transcripts provided.
        You do not have access to any information beyond the video transcript.
        If the question is beyond the scope of the video transcript,
        reply as "Sorry! The question is out of the current context".
        Do not attempt to answer questions that require knowledge beyond the video transcript.

        Here is the question: {question}
        Here is the video transcript: {docs}

        Your answer should be as detailed as possible, using only the information from the video transcript.
        Do not mention the words "Video transcript" in your response.
        """,
    )

    # Create and run the LLMChain to generate the answer
    chain = LLMChain(llm=llm, prompt=prompt)
    response = chain.run(question=query, docs=page_content)
    response = response.replace("\n", "")
    return response

if __name__ == "__main__":
    st.set_page_config(layout="wide")
    
    st.markdown("""
                    <style>
                    @import url('https://fonts.googleapis.com/css2?family=Agdasima');
                    .custom-text-01 { font-family: 'Agdasima', sans-serif; font-size: 60px; color: cyan; }
                    </style>
                    <p class="custom-text-01">Youtube assistant using Langchain and Google Gemini</p>
                    """, unsafe_allow_html=True)
    st.divider()

    st.markdown(
        '<div style="text-align: justify">The Streamlit application “Youtube Assistant using Langchain and Google Gemini” is a sophisticated tool that leverages machine learning and natural language processing to enhance your interaction with YouTube content. The application accepts a YouTube URL and a user question as inputs and uses similarity search on a vector database containing embeddings to provide relevant answers. This is achieved by transforming the content of the YouTube video into a searchable vector database using Langchain Community, Langchain Core, and Google Gemini technologies.</div>',
        unsafe_allow_html=True)
    st.write('')
    st.markdown(
        '<div style="text-align: justify">The application begins by loading the transcript of the provided YouTube video using a custom loader that leverages the youtube_transcript_api. The loaded transcript is then split into manageable chunks using the <code>RecursiveCharacterTextSplitter</code> class, and these chunks are transformed into vector embeddings using the <code>GoogleGenerativeAIEmbeddings</code> class. These embeddings are stored in a FAISS vector database.</div>',
        unsafe_allow_html=True)
    st.write('')
    st.markdown(
        '<div style="text-align: justify">When a user poses a question, the application performs a similarity search on the vector database to identify the most relevant information to answer the query. The application uses the <code>LLMChain</code> class with Google Gemini to generate a detailed response based on the user’s question and the content of the video transcript. If the question is beyond the scope of the transcript, the application will respond with “Sorry! The question is out of the current context”.</div>',
        unsafe_allow_html=True)
    st.write('')
    st.markdown(
        '<div style="text-align: justify">This tool is perfect for extracting specific information from YouTube videos without watching the entire video. Whether it’s a lecture, tutorial, or documentary, “Youtube Assistant using Langchain and Google Gemini” streamlines the process of finding the answers you need – like having a personal assistant for your YouTube viewing experience!</div>',
        unsafe_allow_html=True)
    st.write('')
    st.divider()

    url = st.text_input('Youtube URL', 'Please enter here..', key="url_input")
    st.write('')
    que = st.text_input('Your Question', 'Please enter here..', key="question_input")
    st.write('')
    st.write('')

    if st.button('Fetch the answer..', use_container_width=True):
        if not url:
            st.warning("Please enter a YouTube URL.")
        elif not que:
            st.warning("Please enter your question.")
        else:
            with st.spinner("Loading transcript and creating embeddings..."):
                db = vdb_from_url(url)
            if db:
                with st.spinner("Searching for the answer..."):
                    response = response_for_query(db, que, 4)
                    wrapper = textwrap.TextWrapper(width=100)
                    wrapped_text = wrapper.fill(response)
                    st.write('')
                    st.write('')
                    st.markdown("### Answer:")
                    st.write(wrapped_text)
