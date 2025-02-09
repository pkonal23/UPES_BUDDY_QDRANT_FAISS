from langchain.vectorstores import FAISS
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
from langchain.callbacks import StdOutCallbackHandler
from langchain import PromptTemplate
import os
from langchain_pinecone import PineconeVectorStore
from langchain.embeddings import OpenAIEmbeddings

from dotenv import load_dotenv
load_dotenv()

os.environ["OPENAI_API_KEY"]=os.getenv("OPENAI_API_KEY")
os.environ["LANGCHAIN_TRACING_V2"]="false"
os.environ["LANGCHAIN_API_KEY"]=os.getenv("LANGCHAIN_API_KEY")


llm = ChatOpenAI(model_name='gpt-4o-mini', temperature=0.5)

embeddings_model = OpenAIEmbeddings(model="text-embedding-3-small")
index_name = "upes"
vectordb_file_path = "VectorDB/db_faiss"



def get_qa_chain():
    # Load the vector database from the local folder
    #vectordb = FAISS.load_local(vectordb_file_path, embeddings_model,allow_dangerous_deserialization=True)
    vectordb = PineconeVectorStore(index_name=index_name, embedding=embeddings_model)


    # Create a retriever for querying the vector database
    retriever = vectordb.as_retriever(score_threshold=1, top_k=3)

    prompt_template = """You are UPESBuddy, a virtual assistant created specifically for students of the University of Petroleum and Energy Studies (UPES), Dehradun. Your primary role is to answer questions related to UPES using only the context provided to you, which includes information pertaining to academic programs, admission processes, campus facilities, event schedules, and university policies.

Please adhere to the following guidelines:

1. Respond accurately to user inquiries by referencing only the available context related to UPES. Avoid referencing external information or creating answers based on general knowledge.
2. If the context does not contain the necessary information to answer a user's question, do not attempt to infer an answer or provide speculative responses. Instead, kindly inform the user with the following response:
   "I'm sorry, but I currently do not have the information required to answer your question. I encourage you to consult the official UPES resources or provide additional context."
3. Maintain a courteous and professional tone at all times to ensure a positive and respectful interaction with users.
4. Read the CONTEXT given here carefully to form answer for the QUESTION given after it.
5. If query is about a person and context has information about multiple people with same name then write a short summary with their contact.
----------------------

    - CONTEXT: 
    {context}

--------------------------

    - QUESTION: 
    {question}"""

    PROMPT = PromptTemplate(
        template=prompt_template, input_variables=["context", "question"]
    )

    handler = StdOutCallbackHandler()

    chain = RetrievalQA.from_chain_type(llm=llm,
                                        chain_type="stuff",
                                        retriever=retriever,
                                        input_key="query",
                                        callbacks=[handler],
                                        return_source_documents=True,
                                        chain_type_kwargs={"prompt": PROMPT}, )

    return chain


if __name__ == "__main__":
    chain = get_qa_chain()