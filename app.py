import os
import streamlit as st
from langchain_ollama.llms import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.document_loaders import PyPDFLoader

llm = OllamaLLM(model='mistral')
output_parser = StrOutputParser()


resume_summary_prompt = ChatPromptTemplate([
    ('system', 'You are a helpful assistant named {name}. '
               'When summarizing, start your response with: '
               '"{name}, the helpful assistant, has reviewed your request and here is the summary."'),
    ('user', 'Analyze the attached resume and summarize it: {resume}')
])

resume_match_prompt = ChatPromptTemplate([
    ('system', 'You have to understand the client requirement and based on the job role asked by the client, '
               'analyze the resume and check if the requirements match.'),
    ('user', 'Analyze the resume and match the {requirements} with {resume} and tell user what skills what '
             'they dont have to be eligible for this job work. Also give a short and concise output')
])

# Create AI processing chains
summary_chain = resume_summary_prompt | llm | output_parser
matching_chain = resume_match_prompt | llm | output_parser


st.title("AI-Powered Resume Analyzer")

def process_resume(uploaded_file, analysis_chain, extra_params=None):
    if uploaded_file is None:
        return None

    temp_file_path = "temp_resume.pdf"

    with open(temp_file_path, "wb") as f:
        f.write(uploaded_file.read())

    loader = PyPDFLoader(temp_file_path)
    documents = loader.load()
    resume_text = '\n'.join([doc.page_content for doc in documents])


    input_params = {'resume': resume_text}
    if extra_params:
        input_params.update(extra_params)

    result = analysis_chain.invoke(input_params)


    # os.remove(temp_file_path)

    return result


st.subheader("Resume Summarization")
uploaded_resume = st.file_uploader("Upload your resume for AI summarization.")

if uploaded_resume:
    summary = process_resume(uploaded_resume, summary_chain, {'name': 'Khushi'})
    if summary:
        st.write(" Resume Summary:")
        st.write(summary)

st.subheader("Resume Matching with Job Requirements")
uploaded_resume_match = st.file_uploader("Upload a resume to check job match.")
client_requirements = st.text_input("Enter the job requirements.")

if uploaded_resume_match and client_requirements:
    match_result = process_resume(uploaded_resume_match, matching_chain, {'requirements': client_requirements})
    if match_result:
        st.write("Matching Result:")
        st.write(match_result)
