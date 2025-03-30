import traceback
from urllib.parse import urlparse, parse_qs

import requests
from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
import textwrap
import streamlit as st
import google.generativeai as genai
from youtube_transcript_api import YouTubeTranscriptApi

# Configure Google Generative AI
if "GOOGLE_API_KEY" not in st.secrets:
    st.error("Google API Key not found. Please add it to your Streamlit secrets.")
    st.stop()
genai.configure(api_key=st.secrets["GOOGLE_API_KEY"])

def get_video_id(url):
    parsed = urlparse(url)
    query = parse_qs(parsed.query)
    video_id = query.get("v", [None])[0]
    if not video_id and parsed.hostname in ["youtu.be"]:
        video_id = parsed.path.lstrip("/")
    return video_id

def load_youtube_transcript(url, session):
    video_id = get_video_id(url)
    if not video_id:
        raise ValueError("Invalid YouTube URL; unable to extract video ID.")
    # Instead of using the static method, instantiate the API with a custom session.
    ytt_api = YouTubeTranscriptApi(http_client=session)
    transcript = ytt_api.fetch(video_id)
    # Combine transcript snippets into one string.
    full_text = " ".join([segment["text"] for segment in transcript.to_raw_data()])
    return Document(page_content=full_text, metadata={"video_id": video_id})

def vdb_from_url(url):
    try:
        # Configure your proxy settings here.
        # For example, if using a Tor proxy running on localhost:9050:
        proxies = {
            'http': 'socks5://127.0.0.1:9050',
            'https': 'socks5://127.0.0.1:9050',
        }
        # Create a requests Session and update its proxies.
        session = requests.Session()
        session.proxies.update(proxies)
        # Optionally, disable keep-alive to force new connections.
        session.headers.update({'Connection': 'close'})

        document = load_youtube_transcript(url, session)
        if not document:
            st.warning("Transcript loaded but empty.")
            return None

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        docs_splitted = text_splitter.split_documents([document])
        if not docs_splitted:
            st.warning("Transcript loaded but could not be split into chunks.")
            return None

        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        db = FAISS.from_documents(docs_splitted, embeddings)
        return db

    except Exception as e:
        st.error(f"Error processing YouTube URL: {e}")
        print("Detailed traceback:", traceback.format_exc())
        return None

def response_for_query(db, query, k):
    docs = db.similarity_search(query, k=k)
    page_content = " ".join([d.page_content for d in docs])
    llm = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.0)
    prompt = PromptTemplate(
        input_variables=["question", "docs"],
        template="""
        You are a helpful YouTube assistant powered by Google AI.
        Your mission is to answer questions about videos using only the provided transcript.
        If the question is out of scope, reply with "Sorry! The question is out of the current context".
        Here is the question: {question}
        Here is the transcript: {docs}
        Provide a detailed answer.
        """,
    )
    chain = LLMChain(llm=llm, prompt=prompt)
    response = chain.run(question=query, docs=page_content)
    return response.replace("\n", "")

if __name__ == "__main__":
    st.set_page_config(layout="wide")
    st.markdown("""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Agdasima');
        .custom-text { font-family: 'Agdasima', sans-serif; font-size: 60px; color: cyan; }
        </style>
        <p class="custom-text">Youtube Assistant using Langchain and Google Gemini</p>
    """, unsafe_allow_html=True)
    st.divider()
    st.markdown("This tool extracts YouTube transcripts (using a proxy to avoid IP bans) and answers your questions about the video.")
    url = st.text_input('YouTube URL', 'Enter URL here...', key="url_input")
    que = st.text_input('Your Question', 'Enter your question here...', key="question_input")
    if st.button('Fetch the answer'):
        if not url:
            st.warning("Please enter a YouTube URL.")
        elif not que:
            st.warning("Please enter a question.")
        else:
            with st.spinner("Loading transcript and creating embeddings..."):
                db = vdb_from_url(url)
            if db:
                with st.spinner("Searching for the answer..."):
                    response = response_for_query(db, que, 4)
                    wrapped = textwrap.fill(response, width=100)
                    st.markdown("### Answer:")
                    st.write(wrapped)
