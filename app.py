from langchain_ollama.llms import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.document_loaders import PyPDFLoader
import os
import streamlit as st

template = ChatPromptTemplate([
    ('system', 'You are a helpful assistant named {name}. When summarizing, start your response with: '
               '"{name}, the helpful assistant, has reviewed your request and here is the summary."'),
    ('user', 'Analyze the attached resume and summarize it: {resume}')
])

st.title('AI resume Analyzer')
input_file = st.file_uploader('Hey I am Khushi, You`re AI powered resume analyzer. Upload your resume here.')

llm = OllamaLLM(model='mistral')
output_parser = StrOutputParser()
chain = template | llm | output_parser

if input_file:
    with open('temp_resume.pdf' , 'wb') as f:
        f.write(input_file.read())
    loader = PyPDFLoader('temp_resume.pdf')
    document = loader.load()
    resume_text = '\n'.join([doc.page_content for doc in document])
    result = chain.invoke({'resume':resume_text,
                           'name':'khushi'})
    st.write("### Resume Summary:")
    st.write(result)

    # Clean up the temporary file
    os.remove("temp_resume.pdf")