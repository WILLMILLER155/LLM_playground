from langchain.document_loaders import TextLoader
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.llms import OpenAI
from langchain.chains import RetrievalQA

class RAGPipeline:
    def __init__(self, doc_path, embedding_model_name, llm_model_name):
        self.doc_path = doc_path
        self.embedding_model_name = embedding_model_name
        self.llm_model_name = llm_model_name
        self.vectorstore = None
        self.qa_chain = None
        self._setup_pipeline()
    
    def _setup_pipeline(self):
        # Load documents
        doc_loader = TextLoader(self.doc_path)
        documents = doc_loader.load()
        
        # Embed documents
        embedding_model = HuggingFaceEmbeddings(model_name=self.embedding_model_name)
        self.vectorstore = FAISS.from_documents(documents, embedding_model)
        
        # Initialize LLM
        llm = OpenAI(self.llm_model_name)
        
        # Create Retrieval-Augmented Generation (RAG) pipeline
        self.qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=self.vectorstore.as_retriever()
        )
    
    def query(self, user_query):
        return self.qa_chain.run(user_query)

# Example usage
rag_pipeline = RAGPipeline(
    doc_path="enter path",
    embedding_model_name="enter name",
    llm_model_name="enter name"
)

query = "Enter query"
response = rag_pipeline.query(query)
print(response)