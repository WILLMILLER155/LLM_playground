import boto3
import sagemaker
from sagemaker.pytorch import PyTorch
from sagemaker.model import Model
from sagemaker.predictor import Predictor
from sagemaker.serializers import JSONSerializer
from sagemaker.deserializers import JSONDeserializer
import os

# Set AWS config
aws_region = "region"  # Change as needed
s3_bucket = "bucket endpoint"
role = "set your role"
sess = sagemaker.Session()

# Define S3 paths for data & model output
s3_train_data = f"s3://{s3_bucket}/train/"
s3_model_output = f"s3://{s3_bucket}/output/"

# Fine-Tuning LLaMA Model
llama_estimator = PyTorch(
    entry_point="train.py",  # Your training script
    source_dir="./", #Change as needed
    role=role,
    instance_type="change",  # GPU instance for fine-tuning
    instance_count=1, #Change as needed
    framework_version="change", #Change as needed
    py_version="py38",

    hyperparameters={
        "epochs": x,
        "batch_size": x,
        "learning_rate": x
    },
    output_path=s3_model_output,
    code_location=s3_model_output,
)

# Train the model
llama_estimator.fit({"train": s3_train_data})

# Deploy LLaMA as a SageMaker endpoint
llama_model = Model(
    image_uri="change",
    model_data=llama_estimator.model_data,
    role=role,
    predictor_cls=Predictor,
)

llama_predictor = llama_model.deploy(
    initial_instance_count=1,
    instance_type="change",
    serializer=JSONSerializer(),
    deserializer=JSONDeserializer()
)

print("Model deployed at:", llama_predictor.endpoint_name)

# AWS Lambda Function for Serverless Inference
lambda_client = boto3.client("lambda")
function_name = "llama-inference"


lambda_response = lambda_client.create_function(
    FunctionName=function_name,
    Runtime="python3.8",
    Role=role,
    Handler="lambda_function.lambda_handler",
    Code={"ZipFile": open("lambda_function.zip", "rb").read()},
    Timeout=30,
    MemorySize=512,
    Environment={
        "Variables": {
            "ENDPOINT_NAME": llama_predictor.endpoint_name
        }
    }
)

print("AWS Lambda function deployed for inference.")