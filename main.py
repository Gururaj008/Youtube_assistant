from langchain.document_loaders import YoutubeLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
#from langchain.llms import OpenAI # replaced with Google LLM
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.vectorstores import FAISS
#from langchain.embeddings.openai import OpenAIEmbeddings # replaced with Google Embeddings
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI # importing Google LLM
import textwrap
import streamlit as st
import os

def vdb_from_url(url): # removed openai_api_key argument
  #os.environ["OPENAI_API_KEY"] = openai_api_key # No need for OpenAI API key anymore
  loader = YoutubeLoader.from_youtube_url(url)
  transcript = loader.load()
  #embeddings = OpenAIEmbeddings() # replaced with Google Embeddings
  embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001") # using Google Embeddings
  text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
  docs = text_splitter.split_documents(transcript)
  db = FAISS.from_documents(docs, embeddings)
  return db

def response_for_query(db,query, k):
  docs = db.similarity_search(query, k = k)
  page_content = " ".join([d.page_content for d in docs])
  #llm = OpenAI(model = "text-davinci-003") # replaced with Google LLM
  llm = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0) # using Google LLM (gemini-pro)
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
  chain = LLMChain(llm = llm, prompt = prompt)
  response = chain.run(question = query, docs = page_content)
  response = response.replace("\n", "")
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
    st.markdown('<div style="text-align: justify">The Streamlit application “Youtube Assistant using Langchain and Google Gemini” is a sophisticated tool that leverages the power of machine learning and natural language processing to enhance the user’s interaction with YouTube content. The application accepts a YouTube URL and a user question as inputs, and uses similarity search on a vector database containing embeddings to provide relevant answers. This is achieved by transforming the content of the YouTube video into a searchable vector database using Langchain and Google Gemini technologies.</div>', unsafe_allow_html=True)
    st.write('')
    st.markdown('<div style="text-align: justify">The application begins by loading the transcript of the provided YouTube video using the YoutubeLoader class. The transcript is then split into manageable chunks using the RecursiveCharacterTextSplitter class, and these chunks are transformed into vector embeddings using the GoogleGenerativeAIEmbeddings class. These embeddings are stored in a FAISS vector database. </div>', unsafe_allow_html=True)
    st.write('')
    st.markdown('<div style="text-align: justify">When a user poses a question, the application performs a similarity search on the vector database to identify the most relevant information to answer the user’s query. The application uses the LLMChain class with Google Gemini to generate a detailed response based on the user’s question and the content of the video transcript. If the question is beyond the scope of the video transcript, the application will respond with “Sorry! The question is out of the current context”. </div>', unsafe_allow_html=True)
    st.write('')
    st.markdown('<div style="text-align: justify">This application serves as a powerful tool for users who wish to extract specific information from YouTube content without having to watch the entire video. Whether it’s a lecture, tutorial, documentary, or any other informational content, the “Youtube Assistant using Langchain and Google Gemini” streamlines the process of finding the answers you need. It’s like having a personal assistant for your YouTube viewing experience! </div>', unsafe_allow_html=True)
    st.write('')
    st.divider()
    #open_api_key = st.text_input('OPENAI API Key','Please enter here..',key=1) # Removed OpenAI API Key input
    #st.write('') # Removed OpenAI API Key input
    url = st.text_input('Youtube URL','Please enter here..',key=2)
    st.write('')
    que = st.text_input('Your Question','Please enter here..',key=3)
    st.write('')
    st.write('')
    if st.button('Fetch the answer..', use_container_width=True):
       db = vdb_from_url(url) # removed open_api_key argument
       response= response_for_query(db,que,4)
       wrapper = textwrap.TextWrapper(width=100)
       wrapped_text = wrapper.fill(response)
       st.write('')
       st.write('')
       st.write(wrapped_text)
