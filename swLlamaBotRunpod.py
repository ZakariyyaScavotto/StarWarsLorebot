from torch import cuda, bfloat16
import transformers
import dotenv, os
import torch
from transformers import StoppingCriteria, StoppingCriteriaList
from langchain_huggingface import HuggingFacePipeline
from langchain_unstructured import UnstructuredLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain
import re
from functools import lru_cache
import faiss
import numpy as np
from langchain_community.docstore import InMemoryDocstore
from langchain_community.vectorstores.faiss import FAISS as LangchainFAISS
import logging

class swLlamaBot:
    def __init__(self, model_id='/runpod-volume/models/Llama-2-13b-chat-hf', dataPath="Data/CompiledALLInfo.txt", vecStorePath=None, loadVecStore=False):
        # Setup logging
        self.logger = logging.getLogger(__name__)
        logging.basicConfig(level=logging.INFO)
        # Load environment variables
        dotenv.load_dotenv()
        self.hf_auth = os.getenv('HF_AUTH_TOKEN')
        self.device = f'cuda:{cuda.current_device()}' if cuda.is_available() else 'cpu'
        self.model_id = model_id

        # Initialize tokenizer
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(self.model_id)

        # Define stopping criteria
        self.stop_list = ['\nHuman:', '\n```\n']

        # Initialize pipeline
        self.gen_text = transformers.pipeline(
            model=self.init_model(),
            tokenizer=self.tokenizer,
            return_full_text=True,
            task='text-generation',
            stopping_criteria=self.init_stop_criteria(),
            temperature=0.7,
            max_new_tokens=512,
            repetition_penalty=1.15
        )
        self.logger.info("Pipeline initialized.")

        # Initialize Vecstore and embeddings
        model_name = "/runpod-volume/models/sentence-transformers-all-mpnet-base-v2"
        model_kwargs = {"device": "cuda"}
        self.embeddings = HuggingFaceEmbeddings(model_name=model_name, model_kwargs=model_kwargs)

        if loadVecStore and vecStorePath:
            self.vecStore = FAISS.load_local(vecStorePath, self.embeddings, allow_dangerous_deserialization=True)
            self.validate_vec_store()
        elif not loadVecStore:
            self.vecStore = self.createVecStore(dataPath, vecStorePath)
        else:
            raise ValueError("vecStorePath must be provided if loadVecStore is True")

        self.logger.info("VecStore initialized.")

        # Initialize ConversationalRetrievalChain
        self.chain = ConversationalRetrievalChain.from_llm(
            HuggingFacePipeline(pipeline=self.gen_text),
            self.vecStore.as_retriever(search_kwargs={'score_threshold': 0.8}),
            return_source_documents=True
        )

        self.chat_history = []
        self.helpful_answer_pattern = re.compile(r'Helpful Answer:\s*(.*?)(?=\nHelpful Answer:|$)', re.DOTALL)
        self.standalone_question_pattern = re.compile(r"Sure thing! Here's (?:your standalone question|the rephrased version of your follow-up question):\s*(.*)")
        self.logger.info("Bot ready to chat!")


    def chat(self, user_input, chat_history=None):
        self.logger.info(f"User input: {user_input}")
        
        if chat_history is None:
            chat_history = self.chat_history
        
        # Retrieve documents for debugging
        retrieved_docs = self.chain.retriever.get_relevant_documents(user_input)
        self.logger.info(f"Retrieved documents: {[doc.page_content for doc in retrieved_docs]}")
        
        # Generate the reply
        reply = self.chain.invoke({"question": user_input, "chat_history": chat_history})['answer']
        self.logger.info(f"Model raw output: {reply}")
        
        # Find and process 'Helpful Answer'
        matches = self.helpful_answer_pattern.findall(reply)
        if matches:
            reply = matches[-1]
        
        # Handle standalone questions (max of 5 attempts)
        for i in range(5):
            standalone_question_match = self.standalone_question_pattern.match(reply)
            if standalone_question_match or reply.strip().endswith('?'):
                self.logger.info("STANDALONE QUESTION DETECTED")
                standalone_question = standalone_question_match.group(1) if standalone_question_match else reply.strip()
                reply = self.chain.invoke({"question": standalone_question, "chat_history": []})['answer']
                matches = self.helpful_answer_pattern.findall(reply)
                if matches:
                    reply = matches[-1]
            else:
                break
        
        chat_history.append((user_input, reply))
        self.chat_history = chat_history[-5:]
        self.logger.info(f"Final reply: {reply}")
        return reply

    def reset_chat(self):
        self.chat_history = []
        print("Chat history has been reset")
        return "Chat history has been reset"

    @lru_cache(maxsize=1)
    def load_model(self, model_id, hf_auth):
        """
        Loads the model with the given ID and authentication token. 
        This method is cached using lru_cache to ensure the model is loaded only once, avoiding redundant loading operations.
        """
        # set quantization configuration to load large model with less GPU memory
        bnb_config = transformers.BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type='nf4',
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=bfloat16
        )
        model_config = transformers.AutoConfig.from_pretrained(model_id, token=hf_auth)
        model = transformers.AutoModelForCausalLM.from_pretrained(
            model_id,
            trust_remote_code=True,
            config=model_config,
            quantization_config=bnb_config,
            device_map='auto',
            token=hf_auth
        )
        model.eval()
        return model

    def init_model(self):
        model = self.load_model(self.model_id, self.hf_auth)
        print(f"Model loaded on {self.device}")
        return model
    
    def init_stop_criteria(self):
        stop_token_ids = [self.tokenizer(x)['input_ids'] for x in self.stop_list]
        stop_token_ids = [torch.LongTensor(x).to(self.device) for x in stop_token_ids]
        stopping_criteria = StoppingCriteriaList([StopOnTokens(stop_token_ids)])
        return stopping_criteria
    
    def validate_vec_store(self):
        if not self.vecStore or not self.vecStore.index:
            self.logger.error("Vector store is not initialized properly!")
            raise ValueError("Vector store is not initialized properly!")
        num_vectors = self.get_num_vectors()
        self.logger.info(f"Vector store initialized with {num_vectors} vectors.")

    def createVecStore(self, dataPath, vecStorePath="FAISSvectorstore", batch_size=1000):
        loader = UnstructuredLoader(dataPath)
        print("Loading data...")
        documents = loader.load()
        print("Data loaded")
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=20)
        all_splits = text_splitter.split_documents(documents)
        print(f"Number of splits: {len(all_splits)}")
        # Prepare the initial batch of embeddings
        initial_batch = all_splits[:batch_size]
        initial_batch_texts = [doc.page_content for doc in initial_batch]
        initial_batch_embeddings = self.embeddings.embed_documents(initial_batch_texts)
        # Create IVFPQ index
        d = len(initial_batch_embeddings[0])  # Dimension of embeddings
        nlist = 100  # Number of clusters (adjust as needed)
        m = 8  # Number of subquantizers (adjust as needed)
        nbits = 8  # Number of bits per code in each subquantizer (usually 8)
        quantizer = faiss.IndexFlatL2(d)  # Flat index used for clustering
        index = faiss.IndexIVFPQ(quantizer, d, nlist, m, nbits)
        # Initialize FAISS index (e.g., IVFFlat)
        # d = len(initial_batch_embeddings[0])  # Dimension of embeddings
        # nlist = 100  # Number of clusters (adjust as needed)
        # quantizer = faiss.IndexFlatL2(d)  # Flat index used for clustering
        # index = faiss.IndexIVFFlat(quantizer, d, nlist, faiss.METRIC_L2)
        # Train the index with initial batch embeddings
        index.train(np.array(initial_batch_embeddings))
        print("FAISS index trained")
        # Add initial batch embeddings to the index
        index.add(np.array(initial_batch_embeddings))
        print("Initial batch added to FAISS index")
        # Batch process the remaining splits and add them to the index
        for i in range(batch_size, len(all_splits), batch_size):
            batch = all_splits[i:i+batch_size]
            # print(f"Processing batch {i//batch_size + 1}")
            # Extract text from Document objects in the batch
            batch_texts = [doc.page_content for doc in batch]
            batch_embeddings = self.embeddings.embed_documents(batch_texts)
            # Add batch embeddings to the index
            index.add(np.array(batch_embeddings))
        print("All splits added to FAISS index")
        # Create a LangChain FAISS vector store
        docstore = InMemoryDocstore({str(i): doc for i, doc in enumerate(all_splits)})
        index_to_docstore_id = {i: str(i) for i in range(len(all_splits))}
        vectorstore = LangchainFAISS(index=index, docstore=docstore, index_to_docstore_id=index_to_docstore_id, embedding_function=self.embeddings.embed_documents)
        print("Vector store created")
        # Save the vector store using LangChain's method
        vectorstore.save_local(vecStorePath)
        print("Vector store saved to file")
        return vectorstore
    
    def check_index_type(self):
        faiss_index = self.vecStore.index 
        index_type = type(faiss_index).__name__
        print(f"FAISS Index Type: {index_type}")
        return index_type
    
    def get_num_vectors(self):
        faiss_index = self.vecStore.index 
        num_vectors = faiss_index.ntotal
        print(f"Number of vectors in the FAISS index: {num_vectors}")
        return num_vectors

class StopOnTokens(StoppingCriteria):
    def __init__(self, stop_token_ids):
        self.stop_token_ids = stop_token_ids

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        for stop_ids in self.stop_token_ids:
            if torch.eq(input_ids[0][-len(stop_ids):], stop_ids).all():
                return True
        return False