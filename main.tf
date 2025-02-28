// Terraform script to automate aws provision // 

provider "aws" {
  region = "us-east-1"
}

resource "aws_s3_bucket" "llama_bucket" {
  bucket = "your bucket
  acl    = "private"
}

resource "aws_iam_role" "sagemaker_role" {
  name = "input name"
  assume_role_policy = jsonencode({
    Statement = [{
      Action = "sts:AssumeRole",
      Effect = "Allow",
      Principal = { Service = "sagemaker.amazonaws.com" }
    }]
  })
}

resource "aws_sagemaker_model" "llama_model" {
  name          = "llama-fine-tuned"
  execution_role_arn = aws_iam_role.sagemaker_role.arn
  primary_container {
    image        = "enter"
    model_data_url = "enter"
  }
}

resource "aws_sagemaker_endpoint" "llama_endpoint" {
  name          = "llama-endpoint"
  endpoint_config_name = aws_sagemaker_model.llama_model.name
}