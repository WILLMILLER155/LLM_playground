# <center> LLM_playground </center>

## AWS_Deploy 
This script it to test around AWS configuration and deploying and finetuning an LLM on sagemaker and its endpoints as well.
The general idea can be ported over to Azure, or GCP. This helped me get up to speed on cloud tools and technologies. Used Lambda an AWS tool for serverless inference.
Could have also uploaded a docker file as well with exposed endpoints hosted in a cloud as well.

## Main.tf
This is a terraform script that helped me learn houw to automate provisioning resources on the AWS cloud, this was a starting point for me to learn IaaS for Azure as well.
  + Variables.tf & output.tf
      - These are both terraform scripts to go along with the main terraform script above

## Simple Langchain
Just as the name implies, its a simple structure for a Rag pipline using FAISS vector database, small sample of documents so in memory vector store was great, simple and fast.

## Train.py
This is a script to help train an LLM model along with retriving the embeddings model. Something simple and easy to run through but usually the base of much larger projects.

# ALL SCRIPTS RESIDE IN THE MASTER BRANCH THE README IS IN THE MAIN BRANCH
