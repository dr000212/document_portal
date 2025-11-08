import sys
import os
import streamlit as st
from dotenv import load_dotenv
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_community.vectorstores import FAISS
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from utils.LLM_Loader import ModelLoader
from exception.custom_exception import DocumentPortalException
from logger.custom_logging import CustomLogger
from prompt_template.prompt import PROMPT_REGISTRY
from model.model import PromptType

load_dotenv()


class ConversationalRAG:
    def __init__(self, session_id: str, retriever):
        self.log = CustomLogger().get_logger(__name__)
        self.session_id = session_id
        self.retriever = retriever

        try:
            self.llm = self._load_llm()
            self.contextualize_prompt = PROMPT_REGISTRY[PromptType.CONTEXTUALIZE_QUESTION.value]
            self.qa_prompt = PROMPT_REGISTRY[PromptType.CONTEXT_QA.value]

            self.history_aware_retriever = create_history_aware_retriever(
                self.llm, self.retriever, self.contextualize_prompt
            )
            self.log.info(f"Created history-aware retriever | session_id={session_id}")

            self.qa_chain = create_stuff_documents_chain(self.llm, self.qa_prompt)
            self.rag_chain = create_retrieval_chain(self.history_aware_retriever, self.qa_chain)
            self.log.info(f"Created RAG chain | session_id={session_id}")

            self.chain = RunnableWithMessageHistory(
                self.rag_chain,
                self._get_session_history,
                input_messages_key="input",
                history_messages_key="chat_history",
                output_messages_key="answer"
            )
            self.log.info(f"Wrapped chain with message history | session_id={session_id}")

        except Exception as e:
            self.log.error(f"Error initializing ConversationalRAG | session_id={session_id} | error={e}")
            raise DocumentPortalException("Failed to initialize ConversationalRAG", sys)

    def _load_llm(self):
        try:
            llm = ModelLoader().load_llm()
            self.log.info(f"LLM loaded successfully | class_name={llm.__class__.__name__}")
            return llm
        except Exception as e:
            self.log.error(f"Error loading LLM via ModelLoader: {e}")
            raise DocumentPortalException("Failed to load LLM", sys)

    def _get_session_history(self, session_id: str) -> BaseChatMessageHistory:
        try:
            # If running inside Streamlit, use the real session_state
            if hasattr(st, "session_state"):
                store = st.session_state
            else:
                # Fallback to local store if Streamlit is not running
                if not hasattr(self, "_local_store"):
                    self._local_store = {}
                store = self._local_store

            # Handle both dict-based and attribute-based session_state
            if isinstance(store, dict):
                if "store" not in store:
                    store["store"] = {}
                session_store = store["store"]
            else:
                if not hasattr(store, "store"):
                    store.store = {}
                session_store = store.store

            if session_id not in session_store:
                session_store[session_id] = ChatMessageHistory()
                self.log.info(f"New chat session history created | session_id={session_id}")

            return session_store[session_id]

        except Exception as e:
            self.log.error(f"Failed to access session history | session_id={session_id} | error={e}")
            raise DocumentPortalException("Failed to retrieve session history", sys)


    def load_retriever_from_faiss(self, index_path: str):
        try:
            embeddings = ModelLoader().load_embeddings()
            if not os.path.isdir(index_path):
                raise FileNotFoundError(f"FAISS index directory not found: {index_path}")

            vectorstore = FAISS.load_local(index_path, embeddings)
            self.log.info(f"Loaded retriever from FAISS index | path={index_path}")
            return vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 5})

        except Exception as e:
            self.log.error(f"Failed to load retriever from FAISS: {e}")
            raise DocumentPortalException("Error loading retriever from FAISS", sys)
        
    def invoke(self, user_input: str) -> str:
        try:
            response = self.chain.invoke(
                {"input": user_input},
                config={"configurable": {"session_id": self.session_id}}
            )
            answer = response.get("answer", "No answer.")

            if not answer:
                self.log.warning(f"Empty answer received | session_id={self.session_id}")

            self.log.info(f"Chain invoked successfully | session_id={self.session_id} | input={user_input} | preview={answer[:150]}")
            return answer

        except Exception as e:
            self.log.error(f"Failed to invoke conversational RAG | session_id={self.session_id} | error={e}")
            raise DocumentPortalException("Failed to invoke RAG chain", sys)
