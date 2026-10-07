import os
import streamlit as st
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

from langchain_groq import ChatGroq

from langchain_core.prompts import ChatPromptTemplate


load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

st.set_page_config(
    page_title="RAG Application- PDF Question Answering",
    page_icon="📚"
)

st.title("📚 RAG Application - PDF Question Answering")
st.write("Upload a PDF and ask questions from the document.")

uploaded_file = st.file_uploader(
    "Upload your PDF",
    type=["pdf"]
)

if uploaded_file is not None:
    pdf_path = "temp.pdf"

    with open(pdf_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    st.success("PDF uploaded successfully!")

    loader = PyPDFLoader(pdf_path)
    documents = loader.load()
    st.write(f"📄 Pages loaded: {len(documents)}")

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=20,
        chunk_overlap=5
    )
    chunks = text_splitter.split_documents(documents)

    st.write(f" Number of chunks: {len(chunks)}")

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    vectorstore = FAISS.from_documents(
        chunks,
        embeddings
    )

    st.success(" Vector database created!")

    question = st.text_input(
        "Ask a question about your PDF:"
    )

    if question:
        retrieved_docs = vectorstore.similarity_search(
            question,
            k=5
        )

        context = "\n\n".join(
            [doc.page_content for doc in retrieved_docs]
        )

        llm = ChatGroq(
            model="openai/gpt-oss-20b",
            temperature=0.1
        )

        prompt = ChatPromptTemplate.from_template(
            """
            You are a helpful AI assistant.

            Answer the question using ONLY the
            information provided in the context.

            If the answer is not available in the context,
            say "I don't know based on the document."

            Context:
            {context}

            Question:
            {question}

            Answer:
            """
        )

        final_prompt = prompt.format(
            context=context,
            question=question
        )

        response = llm.invoke(final_prompt)

        st.subheader("Answer")
        st.write(response.content)

        with st.expander("View Retrieved Context"):
            for i, doc in enumerate(retrieved_docs):
                st.write(f"### Retrieved Chunk {i + 1}")
                st.write(doc.page_content)
                st.write(
                    f"Page: {doc.metadata.get('page', 'Unknown') + 1}"
                )
