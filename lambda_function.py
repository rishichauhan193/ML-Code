import json
import os
import boto3
import logging

from langchain.text_splitter import CharacterTextSplitter
from langchain_community.embeddings import FakeEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain_community.llms import FakeListLLM

# Setup logging
logger = logging.getLogger()
logger.setLevel(logging.INFO)

# Initialize LangChain components (cold start optimization)
def setup_qa_chain():
    try:
        knowledge_path = os.path.join(os.path.dirname(__file__), "knowledge.txt")
        with open(knowledge_path, "r") as file:
            data = file.read()

        text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        texts = text_splitter.split_text(data)

        embeddings = FakeEmbeddings(size=1536)
        db = FAISS.from_texts(texts, embeddings)
        retriever = db.as_retriever()

        fake_llm = FakeListLLM(responses=["This is a placeholder response."])
        return RetrievalQA.from_chain_type(llm=fake_llm, chain_type="stuff", retriever=retriever)
    except Exception as e:
        logger.error(f"Failed to set up QA chain: {e}")
        raise

# Setup globally for Lambda cold start reuse
qa = setup_qa_chain()

# Initialize CodePipeline client with region fallback
codepipeline = boto3.client('codepipeline', region_name=os.environ.get("AWS_REGION", "ap-south-1"))

def lambda_handler(event, context):
    logger.info("Received event: %s", json.dumps(event))

    query = event.get("queryStringParameters", {}).get("q", "")
    if not query:
        return {
            "statusCode": 400,
            "body": json.dumps({"error": "Missing query parameter 'q'"})
        }

    try:
        answer = qa.invoke(query)
        logger.info("LangChain answered: %s", answer)

        # Trigger the next pipeline
        response = codepipeline.start_pipeline_execution(name='DeployToCloudFormationService')
        logger.info("Triggered DeployToCloudFormationService: %s", response)

        return {
            "statusCode": 200,
            "body": json.dumps({"answer": answer})
        }

    except codepipeline.exceptions.PipelineNotFoundException:
        error_msg = "Pipeline not found: DeployToCloudFormationService"
        logger.error(error_msg)
        return {
            "statusCode": 500,
            "body": json.dumps({"error": error_msg})
        }
    except Exception as e:
        logger.error("Unhandled exception: %s", str(e))
        return {
            "statusCode": 500,
            "body": json.dumps({"error": str(e)})
        }

# Local test runner
if __name__ == "__main__":
    test_event = {"queryStringParameters": {"q": "What is LangChain?"}}
    print(lambda_handler(test_event, None))
