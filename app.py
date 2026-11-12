
import os
import streamlit as st

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_groq import ChatGroq

from langchain_core.prompts import ChatPromptTemplate
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import (
    create_stuff_documents_chain
)


st.set_page_config(
    page_title="Rainfall Research Assistant",
    page_icon="🌧️",
    layout="wide"
)

st.title("🌧️ Rainfall Research Assistant")
st.caption(
    "Ask questions about Development of Multiple Combined "
    "Regression Methods for Rainfall Measurement."
)


@st.cache_resource
def load_rag():

    db_path = os.path.join(
        os.path.dirname(__file__),
        "rainfall_chroma_db"
    )

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    vectorstore = Chroma(
        persist_directory=db_path,
        collection_name="rainfall_research_v2",
        embedding_function=embeddings
    )

    retriever = vectorstore.as_retriever(
        search_kwargs={"k": 12}
    )

    llm = ChatGroq(
        model="openai/gpt-oss-120b",
        temperature=0.2,
        max_tokens=1200,
        groq_api_key=st.secrets["GROQ_API_KEY"]
    )

    prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            """
            You are a precise research assistant.

            Answer questions using only the supplied research paper
            context.

            Follow these rules:
            1. Read all retrieved context before answering.
            2. Search across all supplied chunks.
            3. Combine table continuation rows when necessary.
            4. Match numerical values to the correct model and metric.
            5. Preserve exact numerical values.
            6. Do not invent facts or experimental results.
            7. If information is unavailable, say so clearly.
            8. If the paper contains conflicting values, explain
               the discrepancy.
            9. Keep answers clear and concise.

            Research context:
            {context}
            """
        ),
        ("human", "{input}")
    ])

    document_chain = create_stuff_documents_chain(
        llm,
        prompt
    )

    rag_chain = create_retrieval_chain(
        retriever,
        document_chain
    )

    return rag_chain


try:
    rag_chain = load_rag()
    st.success("Research database loaded successfully.")

except Exception as e:
    st.error(f"Unable to load the chatbot: {e}")
    st.stop()


with st.sidebar:
    st.subheader("About")
    st.write(
        "This chatbot uses a research paper indexed "
        "with Sentence Transformers and ChromaDB."
    )

    if st.button("Clear conversation"):
        st.session_state.messages = []
        st.rerun()


if "messages" not in st.session_state:
    st.session_state.messages = []


for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])


question = st.chat_input(
    "Ask a question about the rainfall research paper..."
)


if question:

    st.session_state.messages.append({
        "role": "user",
        "content": question
    })

    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):

        with st.spinner("Searching the research paper..."):

            try:
                response = rag_chain.invoke({
                    "input": question
                })

                answer = response["answer"]

                st.markdown(answer)

                source_docs = response.get("context", [])

                if source_docs:

                    with st.expander("View research sources"):

                        for index, doc in enumerate(source_docs, 1):

                            page = doc.metadata.get(
                                "page",
                                "Not specified"
                            )

                            st.markdown(
                                f"**Source {index} — PDF page {page}**"
                            )

                            st.caption(
                                doc.page_content[:400]
                            )

                st.session_state.messages.append({
                    "role": "assistant",
                    "content": answer
                })

            except Exception as e:
                st.error(f"Answer generation failed: {e}")
