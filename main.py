# Corrected Imports based on Deprecation Warnings
from langchain_community.document_loaders import YoutubeLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
# Corrected Imports based on Deprecation Warnings
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
import textwrap
import streamlit as st
import os

# Set Google API Key from Streamlit secrets
# Ensure GOOGLE_API_KEY is set in your Streamlit secrets
if "GOOGLE_API_KEY" not in st.secrets:
    st.error("Google API Key not found. Please add it to your Streamlit secrets.")
    st.stop()

# Configure the library using the key from secrets
import google.generativeai as genai
genai.configure(api_key=st.secrets["GOOGLE_API_KEY"])


def vdb_from_url(url):
    try:
        loader = YoutubeLoader.from_youtube_url(url, add_video_info=True) # add_video_info can be helpful
        # loader.load() returns a list of Document objects
        documents = loader.load()

        if not documents:
            st.warning("Could not load transcript. Please check the URL and ensure the video has transcripts available.")
            return None

        # Use the splitter directly on the documents
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        docs_splitted = text_splitter.split_documents(documents)

        if not docs_splitted:
             st.warning("Transcript loaded but could not be split into chunks.")
             return None

        # Create embeddings
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

        # Create FAISS index from the split Document objects
        db = FAISS.from_documents(docs_splitted, embeddings)
        return db

    except Exception as e:
        st.error(f"Error processing YouTube URL: {e}")
        # Optionally log the full error for debugging
        # print(f"Detailed error in vdb_from_url: {traceback.format_exc()}")
        return None


def response_for_query(db,query, k):
  # Perform similarity search on the FAISS index
  # similarity_search returns Document objects
  docs = db.similarity_search(query, k = k)

  # Extract page_content from the Document objects
  page_content = " ".join([d.page_content for d in docs])

  # Initialize the LLM
  llm = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.0) # Changed temperature to 0.0 for more deterministic output

  # Define the prompt template
  prompt = PromptTemplate(
      input_variables=["question","docs"],
      template = """
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

  # Create and run the LLMChain
  chain = LLMChain(llm = llm, prompt = prompt)
  response = chain.run(question = query, docs = page_content)
  response = response.replace("\n", "") # Remove newlines for cleaner output (optional)
  return response


if __name__ == "__main__":
    st.set_page_config(layout="wide")

    st.markdown("""
                    <style>
                    @import url('https://fonts.googleapis.com/css2?family=Agdasima');
                    .custom-text-01 { font-family: 'Agdasima', sans-serif; font-size: 60px;color:cyan }
                    </style>
                    <p class="custom-text-01">Youtube assistant using Langchain and Google Gemini</p>
                    """, unsafe_allow_html=True)
    st.divider()
    # Updated description to reflect correct library usage
    st.markdown('<div style="text-align: justify">The Streamlit application “Youtube Assistant using Langchain and Google Gemini” is a sophisticated tool that leverages the power of machine learning and natural language processing to enhance the user’s interaction with YouTube content. The application accepts a YouTube URL and a user question as inputs, and uses similarity search on a vector database containing embeddings to provide relevant answers. This is achieved by transforming the content of the YouTube video into a searchable vector database using Langchain Community, Langchain Core, and Google Gemini technologies.</div>', unsafe_allow_html=True)
    st.write('')
    st.markdown('<div style="text-align: justify">The application begins by loading the transcript of the provided YouTube video using the `YoutubeLoader` from `langchain-community`. The loaded documents are then split into manageable chunks using the `RecursiveCharacterTextSplitter` class, and these chunks are transformed into vector embeddings using the `GoogleGenerativeAIEmbeddings` class. These embeddings are stored in a FAISS vector database using `langchain-community`. </div>', unsafe_allow_html=True)
    st.write('')
    st.markdown('<div style="text-align: justify">When a user poses a question, the application performs a similarity search on the vector database to identify the most relevant information to answer the user’s query. The application uses the `LLMChain` class with Google Gemini to generate a detailed response based on the user’s question and the content of the video transcript. If the question is beyond the scope of the video transcript, the application will respond with “Sorry! The question is out of the current context”. </div>', unsafe_allow_html=True)
    st.write('')
    st.markdown('<div style="text-align: justify">This application serves as a powerful tool for users who wish to extract specific information from YouTube content without having to watch the entire video. Whether it’s a lecture, tutorial, documentary, or any other informational content, the “Youtube Assistant using Langchain and Google Gemini” streamlines the process of finding the answers you need. It’s like having a personal assistant for your YouTube viewing experience! </div>', unsafe_allow_html=True)
    st.write('')
    st.divider()

    url = st.text_input('Youtube URL','Please enter here..',key="url_input") # Changed key for clarity
    st.write('')
    que = st.text_input('Your Question','Please enter here..',key="question_input") # Changed key for clarity
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
                    wrapper = textwrap.TextWrapper(width=100) # Adjust width as needed
                    wrapped_text = wrapper.fill(response)
                    st.write('')
                    st.write('')
                    st.markdown("### Answer:") # Use markdown for better formatting potential
                    st.write(wrapped_text)
