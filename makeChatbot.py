!pip3 install langchain # Framework to help make chatbots
!pip install pypdf # Read pdf for text
!pip3 install pinecone-client # Store vector stores (NLP thing) (Need free key, free version is limited)
!pip install python-dotenv # Used for reading environment variables
!pip install openai # OpenAI API (Need to get paid key :( ))
!pip install tiktoken # Used to tokenize text

import os
from dotenv import load_dotenv

from langchain.document_loaders import UnstructuredPDFLoader, OnlinePDFLoader, PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

#got to load this everytime you close colab
loader = PyPDFLoader("/content/Stevens_2020-2021_Academic_Catalog.pdf") # Setup loading the pdf of text we using

"""Load data"""

data = loader.load() # actually load the data

"""Check out the data we just loaded"""

print(f'you have {len(data)} document(s) in your data')
print(f'There are {len(data[30].page_content)} characters in your document')

"""let's check one of them out!"""

data[70]

"""# Now let's create embeddings 🧠"""

from langchain.vectorstores import Chroma, Pinecone
from langchain.embeddings.openai import OpenAIEmbeddings
import pinecone

"""Load in our keys

```
# OPENAI_API_KEY = "sk-..."
```
"""

OPENAI_API_KEY = ""
embeddings = OpenAIEmbeddings(openai_api_key = OPENAI_API_KEY)

pinecone.init (
    api_key='', # pinecone api key
    environment='' # pinecone environment name
)
index_name = "" # name of your pinecone thing

docsearch = Pinecone.from_texts([t.page_content for t in data], embeddings, index_name=index_name)
# Way to do similarity searching
# Compares vector inputted to document vectors to find content related to the question

"""#Query to find where the chatbot is getting its info from 📑"""

query = "What is Steven's mission as an institution?"
#find docs that are similar to this question
docs = docsearch.similarity_search(query)

docs # print the doc related to the question, not the actual response just pulls specific sections

"""# The grand finale ✅"""

from langchain.llms import OpenAI
from langchain.chains.question_answering import load_qa_chain

#define llm
llm = OpenAI(temperature=1, openai_api_key = OPENAI_API_KEY) # temperature controls how much it changes stuff
#define chain
chain = load_qa_chain(llm, chain_type="stuff")

query = "What is the Stevens Honor system?"
#similarity search
docs = docsearch.similarity_search(query)

# run a chain
chain.run(input_documents=docs, question=query) # actually get output