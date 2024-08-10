from torch import cuda, bfloat16
import transformers
import dotenv, os
import torch
from transformers import StoppingCriteria, StoppingCriteriaList
from langchain_huggingface import HuggingFacePipeline
from langchain_community.document_loaders import UnstructuredFileLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain
import re

class swLlamaBot:
    def __init__(self, model_id = 'meta-llama/Llama-2-13b-chat-hf', dataPath = "CompiledALLInfo.txt", vecStorePath = None, loadVecStore = False):
        # Load environment variables
        dotenv.load_dotenv()
        self.hf_auth = os.getenv('HF_AUTH_TOKEN')
        self.device = f'cuda:{cuda.current_device()}' if cuda.is_available() else 'cpu'
        self.model_id = model_id
        # Initialize tokenizer
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(self.model_id, token=self.hf_auth)
        # Define stop_token_ids
        self.stop_list = ['\nHuman:', '\n```\n']
        # Initialize pipeline
        self.gen_text = transformers.pipeline(
            model=self.init_model(),
            tokenizer=self.tokenizer,
            return_full_text=True,  # langchain expects the full text
            task='text-generation',
            # we pass model parameters here too
            stopping_criteria=self.init_stop_criteria(),  # without this model rambles during chat
            temperature=0.1,  # 'randomness' of outputs, 0.0 is the min and 1.0 the max
            max_new_tokens=512,  # max number of tokens to generate in the output
            repetition_penalty=1.1  # without this output begins repeating
        )
        print("Pipeline initialized")
        # Initialize Vecstore and embeddings
        model_name = "sentence-transformers/all-mpnet-base-v2"
        model_kwargs = {"device": "cuda"}
        self.embeddings = HuggingFaceEmbeddings(model_name=model_name, model_kwargs=model_kwargs)
        if loadVecStore and vecStorePath is not None:
            self.vecStore = FAISS.load_local(vecStorePath, self.embeddings, allow_dangerous_deserialization=True)
        elif loadVecStore and vecStorePath is None:
            raise ValueError("vecStorePath must be provided if loadVecStore is True")
        else:
            self.vecStore = self.createVecStore(dataPath)
        print("VecStore initialized")
        # Initialize ConversationalRetrievalChain
        self.chain = ConversationalRetrievalChain.from_llm(HuggingFacePipeline(pipeline=self.gen_text), self.vecStore.as_retriever(), return_source_documents=True)
        # Initialize chat history
        self.chat_history = []
        print("Ready to chat!")

    def chat(self, user_input):
        reply = self.chain.invoke({"question": user_input, "chat_history": self.chat_history})['answer']
        # Find all occurrences of 'Helpful Answer:' and the text that follows
        matches = re.findall(r'Helpful Answer:\s*(.*?)(?=\nHelpful Answer:|$)', reply, re.DOTALL)
        # Select the last match
        if matches:
            reply = matches[-1]  
        # Loop to handle multiple possible standalone questions
        while True:
            # Check if the reply is a standalone question
            standalone_question_match = re.match(r"Sure thing! Here's (?:your standalone question|the rephrased version of your follow-up question|the rephrased version of the follow-up question):\s*(.*)", reply)
            if standalone_question_match or reply.strip().endswith('?'):
                # print("STANDALONE QUESTION DETECTED")
                standalone_question = standalone_question_match.group(1) if standalone_question_match else reply.strip()
                # Invoke the chain again with the standalone question
                reply = self.chain.invoke({"question": standalone_question, "chat_history": []})['answer']
                matches = re.findall(r'Helpful Answer:\s*(.*?)(?=\nHelpful Answer:|$)', reply, re.DOTALL)
                if matches:
                    reply = matches[-1] 
            else:
                break
        self.chat_history.append((user_input, reply))
        print(reply)
        return reply

    def reset_chat(self):
        self.chat_history = []
        print("Chat history has been reset")
        return "Chat history has been reset"

    def init_model(self):
        # set quantization configuration to load large model with less GPU memory
        # this requires the `bitsandbytes` library
        bnb_config = transformers.BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type='nf4',
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=bfloat16
        )
        # begin initializing HF items, you need an access token
        model_config = transformers.AutoConfig.from_pretrained(
            self.model_id,
            token=self.hf_auth
        )
        model = transformers.AutoModelForCausalLM.from_pretrained(
            self.model_id,
            trust_remote_code=True,
            config=model_config,
            quantization_config=bnb_config,
            device_map='auto',
            token=self.hf_auth
        )
        # enable evaluation mode to allow model inference
        model.eval()
        print(f"Model loaded on {self.device}")
        return model
    
    def init_stop_criteria(self):
        stop_token_ids = [self.tokenizer(x)['input_ids'] for x in self.stop_list]
        stop_token_ids = [torch.LongTensor(x).to(self.device) for x in stop_token_ids]
        stopping_criteria = StoppingCriteriaList([StopOnTokens(stop_token_ids)])
        return stopping_criteria
    
    def createVecStore(self, dataPath, vecStorePath = None):
        loader = UnstructuredFileLoader(dataPath)
        documents = loader.load()   
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=20)
        all_splits = text_splitter.split_documents(documents)
        # storing embeddings in the vector store
        vectorstore = FAISS.from_documents(all_splits, self.embeddings)
        # Persisting the vector store
        vectorstore.save_local("FAISSvectorstore")
        return vectorstore

class StopOnTokens(StoppingCriteria):
    def __init__(self, stop_token_ids):
        self.stop_token_ids = stop_token_ids

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        for stop_ids in self.stop_token_ids:
            if torch.eq(input_ids[0][-len(stop_ids):], stop_ids).all():
                return True
        return False