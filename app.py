import streamlit as st
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.chains.summarize import load_summarize_chain
from transformers import T5Tokenizer, T5ForConditionalGeneration
from transformers import pipeline
import torch
import base64

checkpoint = "LaMini-Flan-T5-248M"
checkpoint_folder = "LaMini-Flan-T5-248M"
tokenizer = T5Tokenizer.from_pretrained(checkpoint, legacy=False)
base_model = T5ForConditionalGeneration.from_pretrained(checkpoint, device_map = "auto", offload_folder=checkpoint_folder, torch_dtype = torch.float32)

#File Loader and Preprocessing
def file_processing(file):
  loader = PyPDFLoader(file)
  pages = loader.load_and_split()
  text_splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=50)
  texts = text_splitter.split_documents(pages)
  final_texts = ""
  for text in texts:
    print(text)
    final_texts = final_texts + text.page_content
  return final_texts

#LM pipeline
def LLM_Pipeline(filepath):
  pipeline_summarizer = pipeline(
    'summarization',
    model = base_model,
    tokenizer = tokenizer,
    max_length = 500,
    min_length = 25
  )
  input_pdf = file_processing(filepath)
  result = pipeline_summarizer(input_pdf)
  result = result[0]['summary_text']
  return result

@st.cache_data
#function to display the pdf of a given file
def displayPDF(file):
    # Opening file from file path
    with open(file, "rb") as f:
        base64_pdf = base64.b64encode(f.read()).decode('utf-8')

    # Embedding PDF in HTML
    pdf_display = F'<embed src="data:application/pdf;base64,{base64_pdf}" width="700" height="1000" type="application/pdf">'

    # Displaying File
    st.markdown(pdf_display, unsafe_allow_html=True)

#Streamlit code
st.set_page_config(layout='wide', page_title="PDF Summarizer")

def main():
  st.title('Online PDF Summarizer')
  uploaded_file = st.file_uploader("Upload your PDF File", type=['pdf'])

  if uploaded_file is not None:
    if st.button("Summarize"):
      col1, col2 = st.columns(2)
      filepath = "data/"+uploaded_file.name
      with open(filepath, 'wb') as temp_file:
        temp_file.write(uploaded_file.read())

      with col1:
        st.info("Uploaded PDF File")
        pdf_viewer = displayPDF(filepath)


      with col2:
        st.info("Here is your PDF Summarization")
        summary = LLM_Pipeline(filepath)
        st.success(summary)


if __name__ == "__main__":
  main()
