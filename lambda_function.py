import json
import os
import boto3

from langchain.text_splitter import CharacterTextSplitter
from langchain_community.embeddings import FakeEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain_community.llms import FakeListLLM

# Initialize once (cold start optimization)
def setup_qa_chain():
    knowledge_path = os.path.join(os.path.dirname(__file__), "knowledge.txt")
    with open(knowledge_path, "r") as file:
        data = file.read()

    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    texts = text_splitter.split_text(data)

    embeddings = FakeEmbeddings(size=1536)
    db = FAISS.from_texts(texts, embeddings)
    retriever = db.as_retriever()

    fake_llm = FakeListLLM(responses=["This is a placeholder Response."])
    return RetrievalQA.from_chain_type(llm=fake_llm, chain_type="stuff", retriever=retriever)

qa = setup_qa_chain()
codepipeline = boto3.client('codepipeline')

def lambda_handler(event, context):
    query = event.get("queryStringParameters", {}).get("q", "")
    if not query:
        return {
            "statusCode": 400,
            "body": json.dumps({"error": "Missing query parameter 'q'"})
        }

    try:
        answer = qa.run(query)

        # Trigger the next pipeline
        response = codepipeline.start_pipeline_execution(
            name='DeployToCloudFormationService'
        )
        print("Triggered DeployToCloudFormationService:", response)

        return {
            "statusCode": 200,
            "body": json.dumps({"answer": answer})
        }
    except Exception as e:
        return {
            "statusCode": 500,
            "body": json.dumps({"error": str(e)})
        }

# Local test
if __name__ == "__main__":
    test_event = {"queryStringParameters": {"q": "What is LangChain?"}}
    print(lambda_handler(test_event, None))
