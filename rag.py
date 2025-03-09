import os
import time
import logging
from typing import List, Dict, Any, Optional, Tuple
from functools import lru_cache
import concurrent.futures
from pathlib import Path

import google.generativeai as genai
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain.chains import ConversationalRetrievalChain
from langchain_community.document_loaders import PyPDFLoader, UnstructuredPDFLoader
from langchain_community.vectorstores.utils import filter_complex_metadata
from langchain.memory import ConversationBufferWindowMemory
from langchain.schema import Document
from langchain.prompts import PromptTemplate

from PIL import Image
import io
import base64

from langchain_core.messages import HumanMessage
from langchain_core.messages.system import SystemMessage

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class GemRag:
    def __init__(self, api_key: str, model_name: str = "gemini-1.5-pro-latest", cache_dir: str = "./.cache"):
        start_time = time.time()
        logger.info("Initializing gemrag...")

        genai.configure(api_key=api_key)
        self.model_name = model_name

        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        self.vector_db_path = self.cache_dir / "chroma_db"
        self.vector_db_path.mkdir(exist_ok=True)

        self.llm = ChatGoogleGenerativeAI(
            api_key=api_key,
            model=model_name,
            temperature=0.1,
            max_tokens=1024,
            convert_system_message_to_human=True,
        )

        self.embeddings = GoogleGenerativeAIEmbeddings(
            model="models/embedding-001",
            google_api_key=api_key,
            task_type="RETRIEVAL_QUERY"
        )

        self.memory = ConversationBufferWindowMemory(
            memory_key="chat_history",
            return_messages=True,
            k=5,
        )

        self.vector_store = None
        self.images = []
        self.processed_docs = set()
        self.common_query_terms = {}

        self.system_prompt = """
        You are a high-precision document assistant. Follow these guidelines exactly:
        1. Answer ONLY based on the retrieved document context
        2. If the answer is in the context, provide it with specific citations
        3. If the answer is not in the context, respond with "The documents don't contain this information"
        4. For multimodal elements, describe both text and relevant visuals
        5. Be concise and direct
        Focus exclusively on facts from the documents. Never speculate beyond the provided information.
        """

        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=4)

        logger.info(f"Initialization completed in {time.time() - start_time:.2f} seconds")
    
    def load_pdf(self, file_path: str, chunk_size: int = 2000, chunk_overlap: int = 300, force_reload: bool = False):
        file_path = str(Path(file_path).resolve())
        if file_path in self.processed_docs and not force_reload:
            logger.info(f"Skipping already processed: {file_path}")
            return

        logger.info(f"Processing large PDF: {file_path}")
        
        try:
            with self.executor as executor:
                text_future = executor.submit(self._extract_text, file_path, chunk_size, chunk_overlap)
                image_future = executor.submit(self._extract_images_from_pdf, file_path)
                
                split_docs = text_future.result(timeout=3600)
                extracted_imgs = image_future.result(timeout=3600)

            # Ensure split_docs contains Document objects
            split_docs = [doc for doc in split_docs if isinstance(doc, Document)]

            batch_size = 500
            for i in range(0, len(split_docs), batch_size):
                batch = split_docs[i:i + batch_size]
                if self.vector_store:
                    self.vector_store.add_documents(batch)
                else:
                    self.vector_store = Chroma.from_documents(
                        batch, self.embeddings, persist_directory=str(self.vector_db_path)
                    )
                
                if i % 1000 == 0:
                    self.vector_store.persist()
                    del batch
                    import gc; gc.collect()

            self.images.extend(extracted_imgs)
            self.processed_docs.add(file_path)
            logger.info(f"Processed {len(split_docs)} chunks from {file_path}")

        except Exception as e:
            logger.error(f"Failed processing {file_path}: {str(e)}")
            raise

    def _extract_text(self, file_path: str, chunk_size: int, chunk_overlap: int) -> List[Document]:
        logger.info(f"Extracting text from {Path(file_path).name}")
        
        loader = PyPDFLoader(file_path)
        
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", ". ", "! ", "? ", "; ", ", "],
            keep_separator=True
        )

        try:
            pages = loader.load_and_split()
            docs = []
            
            for i, page in enumerate(pages):
                if not isinstance(page, Document):
                    logger.error(f"Page {i} is not a Document: {type(page)}")
                    continue
                
                chunks = text_splitter.split_documents([page])
                for chunk in chunks:
                    chunk.metadata.update({
                        "source": Path(file_path).name,
                        "page": i + 1,
                        "section": chunk.page_content[:100].strip()
                    })
                docs.extend(chunks)
                
                if i % 100 == 0:
                    import gc; gc.collect()
                    
            return docs
        except Exception as e:
            logger.error(f"Text extraction failed: {str(e)}")
            raise

    def _extract_images_from_pdf(self, file_path: str) -> List[Dict]:
        logger.info(f"Extracting images from {Path(file_path).name}")
        
        try:
            from pdf2image import convert_from_path
            images = convert_from_path(
                file_path,
                dpi=100,
                thread_count=1,  # Explicitly limit to 1 thread
                fmt='jpeg',
                use_pdftocairo=True
            )
            
            extracted_imgs = []
            for i, img in enumerate(images):
                processed_img = self._process_image(img, i, Path(file_path).name)
                if processed_img:
                    extracted_imgs.append(processed_img)
            return extracted_imgs
            
        except Exception as e:
            logger.error(f"Image extraction failed: {str(e)}")
            return []

    def _process_image(self, img, page_num: int, filename: str) -> Dict:
        try:
            img.thumbnail((1200, 1200))
            img_byte_arr = io.BytesIO()
            img.save(img_byte_arr, format='JPEG', quality=85)
            return {
                "image_data": img_byte_arr.getvalue(),
                "page": page_num + 1,
                "source": filename
            }
        except Exception as e:
            logger.error(f"Image processing error: {str(e)}")
            return {}
          
    @lru_cache(maxsize=32)
    def _get_relevant_chunks(self, query: str, k: int = 5) -> List[Document]:
        if not self.vector_store: return []

        return self.vector_store.similarity_search(query, k=k)
    
    def _query_rewriter(self, query: str) -> str:
        filler_words = ["the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for", "with", "about", "is", "are"]
        tokens = query.lower().split()
        
        question_words = ["what", "where", "when", "who", "why", "how"]

        filtered_tokens = [token for token in tokens if token not in filler_words or token in question_words]

        if len(filtered_tokens) < 3 and len(tokens) > 3: return query

        return " ".join(filtered_tokens)

    def setup_rag_chain(self) -> None:
        if self.vector_store is None:
            raise ValueError("No documents loaded. Please load documents first.")
        
        retriever = self.vector_store.as_retriever(
            search_type="similarity_score_threshold",
            search_kwargs={
                "k": 5,
                "score_threshold": 0.7
            }
        )

        qa_prompt_template = """
        CONTEXT INFORMATION:
        {context}
        
        CONVERSATION HISTORY:
        {chat_history}
        
        USER QUESTION: {question}
        
        Provide a concise answer based ONLY on the context information above. 
        If the information isn't in the context, state this clearly.
        """
        qa_prompt = PromptTemplate(
            input_variables=["context", "chat_history", "question"],
            template=qa_prompt_template
        )
        self.chain = ConversationalRetrievalChain.from_llm(
            llm=self.llm,
            retriever=retriever,
            memory=self.memory,
            combine_docs_chain_kwargs={
                "document_prompt": PromptTemplate(
                    input_variables=["page_content"],
                    template="{page_content}"
                ),
                "prompt": qa_prompt
            },
            return_source_documents=True,
            verbose=False
        )

    def search_relevant_images(self, query: str, k: int = 2) -> List[Dict]:
        if not self.images or not self.vector_store: 
            return []
        opt_query = self._query_rewriter(query)
        relevant_docs = self._get_relevant_chunks(opt_query, k=k)
        relevant_pages = {doc.metadata.get("page", 0) for doc in relevant_docs}

        def score_image(img_data):
            page = img_data["page"]
            if page in relevant_pages:
                return 10
            for rel_page in relevant_pages:
                if abs(page - rel_page) == 1: 
                    return 5
                elif abs(page - rel_page) == 2: 
                    return 2
            return 0
        scored_images = [(score_image(img), img) for img in self.images]
        scored_images.sort(key=lambda x: x[0], reverse=True)
        return [img for score, img in scored_images[:k] if score > 0]
    
    def process_query_with_images(self, query: str) -> Tuple[str, List[Dict]]:
        opt_query = self._query_rewriter(query)
        
        start_time = time.time()
        relevant_images = self.search_relevant_images(opt_query)
        text_ctxts = self._get_relevant_chunks(opt_query, k=3)
        text_ctxt = "\n\n".join([doc.page_content for doc in text_ctxts])

        logger.debug(f"Text retrieval completed in {time.time() - start_time:.2f}s")
        sources = []
        seen_sources = set()

        for doc in text_ctxts:
            source = doc.metadata.get("source", "Unknown")
            page = doc.metadata.get("page", "")
            source_info = f"{source} (Page {page})" if page else source
            if source_info not in seen_sources:
                sources.append(source_info)
                seen_sources.add(source_info)
        
        if relevant_images and self.model_name in ["gemini-2.0-flash", "gemini-1.5-pro-latest"]:
            start_time = time.time()
            messages = [
                SystemMessage(content=self.system_prompt),
                HumanMessage(content=[
                    {
                        "type": "text",
                        "text": f"Query: {query}\n\nContext from documents: {text_ctxt}"
                    }
                ])
            ]
            for img_data in relevant_images[:2]:
                pil_image = Image.open(io.BytesIO(img_data["image_data"]))
                img_byte_arr = io.BytesIO()
                pil_image.save(img_byte_arr, format='PNG')
                messages[1].content.append({
                    "type": "image_url",
                    "image_url": f"data:image/png;base64,{base64.b64encode(img_byte_arr.getvalue()).decode('utf-8')}"
                })
            response = self.llm.invoke(messages)
            logger.debug(f"Multimodal processing completed in {time.time() - start_time:.2f}s")
            return response.content, sources
        else:
            response, sources = self.process_query(query)
            return response, sources
        
    def process_query(self, query: str) -> Tuple[str, List[str]]:
        if not hasattr(self, 'chain') or self.chain is None:
            self.setup_rag_chain()
        
        start_time = time.time()
        opt_query = self._query_rewriter(query)
        result = self.chain.invoke({"question": opt_query})
        
        sources = []
        if "source_documents" in result:
            seen_sources = set()
            for doc in result["source_documents"]:
                source = doc.metadata.get("source", "Unknown")
                page = doc.metadata.get("page", "")
                source_info = f"{source} (Page {page})" if page else source
                
                if source_info not in seen_sources:
                    sources.append(source_info)
                    seen_sources.add(source_info)
        
        logger.debug(f"Query processing completed in {time.time() - start_time:.2f}s")
        return result["answer"], sources
    
    def chat(self, user_input: str) -> str:
        if not self.vector_store:
            return "Please load some documents first using the load_pdf method."
        
        start_time = time.time()
        
        if self.images:
            response, sources = self.process_query_with_images(user_input)
        else:
            response, sources = self.process_query(user_input)
            
        formatted_response = response
        if sources:
            formatted_response += "\n\nSources:\n" + "\n".join([f"- {src}" for src in sources])
            
        logger.info(f"Total response time: {time.time() - start_time:.2f}s")
        return formatted_response
    
    def batch_load_pdfs(self, directory_path: str) -> None:
        start_time = time.time()
        directory = Path(directory_path)
        pdf_files = list(directory.glob("*.pdf"))
        
        logger.info(f"Found {len(pdf_files)} PDF files in {directory_path}")
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
            futures = [executor.submit(self.load_pdf, str(pdf_file)) for pdf_file in pdf_files]
            for future in concurrent.futures.as_completed(futures):
                try:
                    future.result()
                except Exception as e:
                    logger.error(f"Error processing PDF: {e}")
        
        logger.info(f"Batch loading completed in {time.time() - start_time:.2f}s")
    
    def optimize_vector_store(self):
        if self.vector_store:
            self.vector_store.persist()
            logger.info("Vector store optimized and persisted")

    def __del__(self):
        if hasattr(self, 'executor'):
            self.executor.shutdown(wait=False)
        if self.vector_store:
            try:
                self.vector_store.persist()
            except Exception as e:
                logger.error(f"Cleanup error: {str(e)}")
                
if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv()
    
    GOOGLE_API_KEY = os.environ.get("GEMINI_API_KEY")
    chatbot = GemRag(api_key=GOOGLE_API_KEY)
    chatbot.load_pdf("./48laws.pdf")
    chatbot.optimize_vector_store()
    while True:
        user_input = input("\nYou: ")
        if user_input.lower() in ["exit", "quit", "bye"]:
            break
        
        response = chatbot.chat(user_input)
        print(f"\nBot: {response}")