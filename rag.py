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
from langchain.vectorstores import Chroma
from langchain.chains import ConversationalRetrievalChain
from langchain.document_loaders import PyPDFLoader, UnstructuredPDFLoader
from langchain.memory import ConversationBufferWindowMemory
from langchain.schema import Document
from langchain.prompts import PromptTemplate

from PIL import Image
import io

from langchain_core.messages import HumanMessage
from langchain_core.messages.system import SystemMessage

# Logger
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)