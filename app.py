import os
import streamlit as st
from langchain_ollama.llms import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.document_loaders import PyPDFLoader
<<<<<<< HEAD
import re

# Enhanced LLM configuration with more detailed prompts
llm = OllamaLLM(model='mistral', temperature=0.3)  # Lower temperature for more focused responses
output_parser = StrOutputParser()

# Refined prompts with more specific instructions
resume_summary_prompt = ChatPromptTemplate.from_messages([
    ('system', 'You are a professional career advisor AI assistant. '
               'Provide a comprehensive yet concise summary of the resume, '
               'focusing     on key professional highlights, skills, and career trajectory. '
               'Structure your summary with clear sections: '
               '1. Professional Profile '
               '2. Key Skills '
               '3. Educational Background '
               '4. Career Achievements'),
    ('user', 'Analyze the following resume and provide a detailed, structured summary:\n{resume}')
])

resume_match_prompt = ChatPromptTemplate.from_messages([
    ('system', 'You are a precise talent matching AI. Conduct a thorough skills gap analysis '
               'by comparing the resume against specific job requirements. Provide a '
               'structured output that clearly identifies: '
               '1. Matching Skills (Strong Alignment) '
               '2. Partial Skills (Need Improvement) '
               '3. Missing Skills (Significant Gaps) '
               '4. Recommendations for Skill Enhancement'),
    ('user', 'Job Requirements: {requirements}\n\nResume Content: {resume}\n\n'
             'Perform a comprehensive skills matching analysis.')
=======

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
>>>>>>> 0e6bc1db729f7a2519bcd847d1fa17eda6ea74ae
])

# Create AI processing chains
summary_chain = resume_summary_prompt | llm | output_parser
matching_chain = resume_match_prompt | llm | output_parser

<<<<<<< HEAD
st.title("Advanced Resume Analysis AI")


def clean_resume_text(text):
    """
    Clean and preprocess resume text to improve analysis quality
    """
    # Remove extra whitespaces
    text = re.sub(r'\s+', ' ', text).strip()

    # Remove special characters and normalize text
    text = re.sub(r'[^\w\s.,\-()]', '', text)

    # Limit text length to prevent overwhelming the model
    return text[:5000]


def process_resume(uploaded_file, analysis_chain, extra_params=None):
    if uploaded_file is None:
        st.error("Please upload a resume PDF")
        return None

    # Secure temp file handling
    try:
        temp_file_path = f"temp_resume_{os.getpid()}.pdf"

        with open(temp_file_path, "wb") as f:
            f.write(uploaded_file.read())

        loader = PyPDFLoader(temp_file_path)
        documents = loader.load()

        # Combine and clean resume text
        resume_text = '\n'.join([doc.page_content for doc in documents])
        cleaned_resume = clean_resume_text(resume_text)

        input_params = {'resume': cleaned_resume}
        if extra_params:
            input_params.update(extra_params)

        result = analysis_chain.invoke(input_params)

    except Exception as e:
        st.error(f"Error processing resume: {e}")
        result = None

    finally:
        # Always attempt to remove temp file
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)
=======

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
>>>>>>> 0e6bc1db729f7a2519bcd847d1fa17eda6ea74ae

    return result


<<<<<<< HEAD
# Streamlit UI Improvements
st.sidebar.header("Resume Analysis Tools")

# Resume Summarization Section
with st.expander("📄 Resume Summarization", expanded=True):
    uploaded_resume = st.file_uploader("Upload Resume for Detailed Analysis", type=['pdf'])

    if uploaded_resume:
        with st.spinner('Analyzing Resume...'):
            summary = process_resume(uploaded_resume, summary_chain)

        if summary:
            st.success("Resume Analysis Complete")
            st.markdown("### 🔍 Detailed Resume Insights")
            st.write(summary)

# Job Matching Section
with st.expander("🎯 Job Requirement Matching", expanded=True):
    uploaded_resume_match = st.file_uploader("Upload Resume for Job Matching", type=['pdf'])
    client_requirements = st.text_area("Paste Detailed Job Requirements")

    if uploaded_resume_match and client_requirements:
        with st.spinner('Matching Resume to Job Requirements...'):
            match_result = process_resume(
                uploaded_resume_match,
                matching_chain,
                {'requirements': client_requirements}
            )

        if match_result:
            st.success("Job Matching Analysis Complete")
            st.markdown("### 📊 Skills Alignment Report")
            st.write(match_result)

# Additional UI Enhancements
st.sidebar.markdown("### 💡 Tips")
st.sidebar.markdown("""
- Upload PDF resumes only
- Provide detailed job requirements
- Ensure clear, readable PDFs
""")
=======
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
>>>>>>> 0e6bc1db729f7a2519bcd847d1fa17eda6ea74ae
