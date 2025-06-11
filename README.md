# ![aws](https://github.com/julien-muke/Search-Engine-Website-using-AWS/assets/110755734/01cd6124-8014-4baa-a5fe-bd227844d263) AI Image Recognition System with AWS Bedrock, Rekognition & Terraform

<div align="center">

  <br />
    <a href="https://youtu.be/63McfqGULvA?si=A7jpVj9SZ1Ad9b9D" target="_blank">
      <img src="https://github.com/user-attachments/assets/fc2fca1b-f0d0-41e6-81c0-67e75942385d" alt="Project Banner">
    </a>
  <br />

<h3 align="center">Serverless Image Analysis on AWS Using Bedrock, Rekognition & IaC with Terraform</h3>

   <div align="center">
     Build this hands-on demo step by step with my detailed tutorial on <a href="http://www.youtube.com/@julienmuke/videos" target="_blank"><b>Julien Muke</b></a> YouTube. Feel free to subscribe 🔔!
    </div>
</div>

## 🚨 Tutorial

This repository contains the steps corresponding to an in-depth tutorial available on my YouTube
channel, <a href="http://www.youtube.com/@julienmuke/videos" target="_blank"><b>Julien Muke</b></a>.

If you prefer visual learning, this is the perfect resource for you. Follow my tutorial to learn how to build projects
like these step-by-step in a beginner-friendly manner!

<a href="https://youtu.be/63McfqGULvA?si=A7jpVj9SZ1Ad9b9D" target="_blank"><img src="https://github.com/sujatagunale/EasyRead/assets/151519281/1736fca5-a031-4854-8c09-bc110e3bc16d" /></a>

## <a name="introduction">🤖 Introduction</a>

Welcome to this exciting hands-on project where we build a complete AI image analysis system on AWS, combining computer vision and generative AI in a fully serverless architecture, all deployed and managed using Terraform.


## <a name="steps">🔎 Overview </a>
 
In this project, we’ll use Amazon Rekognition to detect objects, scenes, and concepts in an image, then pass those results to Amazon Bedrock (Titan model) to generate a human-readable summary. The frontend allows users to upload images and get insightful, AI-generated descriptions, all with zero servers to manage!

## <a name="steps">🛠 Tech Stack: </a>

• Amazon Rekognition – Detects objects, scenes, and labels in images<br>
• Amazon Bedrock (Titan) – Converts labels into descriptive text using generative AI<br>
• AWS Lambda (Python) – Processes requests and orchestrates AI services<br>
• Amazon API Gateway – Exposes our backend via a RESTful API<br>
• Amazon S3 – Hosts a static frontend (HTML/CSS/JS)<br>
• Terraform – Provisions the full infrastructure as code (IaC)<br>

## <a name="pre">📋 Prerequisites </a>

Before you begin, ensure you have the following set up:
 
• **AWS Account**: An active AWS account with administrative privileges to create the necessary resources.<br>
• **AWS CLI**: The AWS Command Line Interface installed and configured with your credentials.<br>
• **Terraform**: Terraform installed on your local machine. You can verify the installation by running `terraform --version`<br>
• **Node.js, npm and Python**: Required for managing frontend dependencies if you choose to expand the project.<br>
• **Model Access in Amazon Bedrock**: You must enable access to the foundation models you intend to use. For this project, navigate to the Amazon Bedrock console, go to Model access, and request access to Titan Image Generator G1.<br>

## ➡️ Step 1 - Project Structure

First, let's organize our project files. Create a main directory for your project, and inside it, create the following structure:

    image-analysis-app/
    ├── terraform/
    │   ├── main.tf
    │   ├── variables.tf
    │   └── outputs.tf
    ├── lambda/
    │   └── image_analyzer.py
    └── frontend/
        ├── index.html
        ├── style.css
        └── script.js

## ➡️ Step 2 - Backend Development with Python and Lambda

We'll start by writing the Python code for our Lambda function. This function will be the brains of our operation.

<details>
<summary><code>lambda/image_analyzer.py</code></summary>

```py
import json
import boto3
import base64

# Initialize AWS clients
rekognition = boto3.client('rekognition')
bedrock_runtime = boto3.client('bedrock-runtime')

def lambda_handler(event, context):
    """
    This Lambda function analyzes an image provided as a base64 encoded string.
    It uses Rekognition to detect labels and Bedrock (Titan) to generate a
    human-readable description.
    """
    try:
        # Get the base64 encoded image from the request body
        body = json.loads(event.get('body', '{}'))
        image_base64 = body.get('image')

        if not image_base64:
            return {
                'statusCode': 400,
                'body': json.dumps({'error': 'No image provided in the request body.'})
            }

        # Decode the base64 string
        image_bytes = base64.b64decode(image_base64)

        # 1. Analyze image with AWS Rekognition
        rekognition_response = rekognition.detect_labels(
            Image={'Bytes': image_bytes},
            MaxLabels=10,
            MinConfidence=80
        )
        labels = [label['Name'] for label in rekognition_response['Labels']]

        if not labels:
             return {
                'statusCode': 200,
                'body': json.dumps({
                    'labels': [],
                    'description': "Could not detect any labels with high confidence. Please try another image."
                })
            }

        # 2. Enhance results with Amazon Bedrock
        # Create a prompt for the Titan model
        prompt = f"Based on the following labels detected in an image: {', '.join(labels)}. Please generate a single, descriptive sentence about the image."

        # Configure the payload for the Bedrock model
        bedrock_payload = {
            "inputText": prompt,
            "textGenerationConfig": {
                "maxTokenCount": 100,
                "stopSequences": [],
                "temperature": 0.7,
                "topP": 0.9
            }
        }

        # Invoke the Bedrock model
        bedrock_response = bedrock_runtime.invoke_model(
            body=json.dumps(bedrock_payload),
            modelId='amazon.titan-text-express-v1',
            contentType='application/json',
            accept='application/json'
        )

        response_body = json.loads(bedrock_response['body'].read())
        description = response_body['results'][0]['outputText'].strip()

        # 3. Return the results
        return {
            'statusCode': 200,
            'headers': {
                'Access-Control-Allow-Origin': '*', # Enable CORS
                'Access-Control-Allow-Headers': 'Content-Type',
                'Access-Control-Allow-Methods': 'OPTIONS,POST'
            },
            'body': json.dumps({
                'labels': labels,
                'description': description
            })
        }

    except Exception as e:
        return {
            'statusCode': 500,
            'body': json.dumps({'error': str(e)})
        }
```
</details>

⚠️Note: This script uses the `boto3` AWS SDK for Python. It will perform the following actions:
1. Receive a base64-encoded image from the API Gateway.
2. Decode the image.
3. Send the image to Amazon Rekognition to detect labels.
4. Create a prompt with these labels and send it to Amazon Bedrock.
5. Return the labels and the AI-generated description.


## ➡️ Step 3 - Infrastructure as Code with Terraform

Now, let's define all the AWS resources needed for our backend using Terraform.

Define some variables to make your configuration reusable.

<details>
<summary><code>terraform/variables.tf</code></summary>

```tf
variable "aws_region" {
  description = "The AWS region to deploy resources in."
  type        = string
  default     = "us-east-1"
}

variable "project_name" {
  description = "A unique name for the project to prefix resources."
  type        = string
  default     = "ai-image-analyzer"
}
```
</details>

This is the main configuration file where we define all our resources.

<details>
<summary><code>terraform/main.tf</code></summary>

```tf
# ==============================================================================
# Provider Configuration
# ==============================================================================
terraform {
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
  }
}

provider "aws" {
  region = var.aws_region
}

# ==============================================================================
# IAM Role and Policies for Lambda
# ==============================================================================
resource "aws_iam_role" "lambda_exec_role" {
  name = "${var.project_name}-lambda-exec-role"

  assume_role_policy = jsonencode({
    Version = "2012-10-17",
    Statement = [{
      Action = "sts:AssumeRole",
      Effect = "Allow",
      Principal = {
        Service = "lambda.amazonaws.com"
      }
    }]
  })
}

resource "aws_iam_policy" "lambda_logging_policy" {
  name        = "${var.project_name}-lambda-logging-policy"
  description = "IAM policy for Lambda to write logs to CloudWatch"

  policy = jsonencode({
    Version = "2012-10-17",
    Statement = [{
      Action = [
        "logs:CreateLogGroup",
        "logs:CreateLogStream",
        "logs:PutLogEvents"
      ],
      Effect   = "Allow",
      Resource = "arn:aws:logs:*:*:*"
    }]
  })
}

resource "aws_iam_policy" "lambda_ai_services_policy" {
  name        = "${var.project_name}-ai-services-policy"
  description = "IAM policy for Lambda to access Rekognition and Bedrock"

  policy = jsonencode({
    Version = "2012-10-17",
    Statement = [
      {
        Action   = "rekognition:DetectLabels",
        Effect   = "Allow",
        Resource = "*"
      },
      {
        Action   = "bedrock:InvokeModel",
        Effect   = "Allow",
        Resource = "arn:aws:bedrock:${var.aws_region}::foundation-model/amazon.titan-text-express-v1"
      }
    ]
  })
}

resource "aws_iam_role_policy_attachment" "lambda_logs_attach" {
  role       = aws_iam_role.lambda_exec_role.name
  policy_arn = aws_iam_policy.lambda_logging_policy.arn
}

resource "aws_iam_role_policy_attachment" "lambda_ai_services_attach" {
  role       = aws_iam_role.lambda_exec_role.name
  policy_arn = aws_iam_policy.lambda_ai_services_policy.arn
}

# ==============================================================================
# Lambda Function
# ==============================================================================
data "archive_file" "lambda_zip" {
  type        = "zip"
  source_dir  = "../lambda/"
  output_path = "${path.module}/image_analyzer.zip"
}

resource "aws_lambda_function" "image_analyzer_lambda" {
  filename      = data.archive_file.lambda_zip.output_path
  function_name = "${var.project_name}-function"
  role          = aws_iam_role.lambda_exec_role.arn
  handler       = "image_analyzer.lambda_handler"
  runtime       = "python3.9"
  timeout       = 30
  source_code_hash = data.archive_file.lambda_zip.output_base64sha256
}

# ==============================================================================
# API Gateway (Simplified for Lambda Proxy)
# ==============================================================================
resource "aws_api_gateway_rest_api" "api" {
  name        = "${var.project_name}-api"
  description = "API for the Image Analyzer"
}

resource "aws_api_gateway_resource" "resource" {
  rest_api_id = aws_api_gateway_rest_api.api.id
  parent_id   = aws_api_gateway_rest_api.api.root_resource_id
  path_part   = "analyze"
}

resource "aws_api_gateway_method" "method" {
  rest_api_id   = aws_api_gateway_rest_api.api.id
  resource_id   = aws_api_gateway_resource.resource.id
  http_method   = "POST"
  authorization = "NONE"
}

resource "aws_api_gateway_integration" "integration" {
  rest_api_id             = aws_api_gateway_rest_api.api.id
  resource_id             = aws_api_gateway_resource.resource.id
  http_method             = aws_api_gateway_method.method.http_method
  integration_http_method = "POST"
  type                    = "AWS_PROXY"
  uri                     = aws_lambda_function.image_analyzer_lambda.invoke_arn
}

# This OPTIONS method is still needed for the browser's preflight request for CORS
resource "aws_api_gateway_method" "options_method" {
  rest_api_id   = aws_api_gateway_rest_api.api.id
  resource_id   = aws_api_gateway_resource.resource.id
  http_method   = "OPTIONS"
  authorization = "NONE"
}

resource "aws_api_gateway_integration" "options_integration" {
  rest_api_id = aws_api_gateway_rest_api.api.id
  resource_id = aws_api_gateway_resource.resource.id
  http_method = aws_api_gateway_method.options_method.http_method
  type        = "MOCK"

  # The MOCK integration returns a success response with the necessary headers.
  request_templates = {
    "application/json" = "{\"statusCode\": 200}"
  }
}

resource "aws_api_gateway_method_response" "options_response" {
  rest_api_id = aws_api_gateway_rest_api.api.id
  resource_id = aws_api_gateway_resource.resource.id
  http_method = aws_api_gateway_method.options_method.http_method
  status_code = "200"

  response_parameters = {
    "method.response.header.Access-Control-Allow-Headers" = true,
    "method.response.header.Access-Control-Allow-Methods" = true,
    "method.response.header.Access-Control-Allow-Origin"  = true
  }
}

resource "aws_api_gateway_integration_response" "options_integration_response" {
  rest_api_id = aws_api_gateway_rest_api.api.id
  resource_id = aws_api_gateway_resource.resource.id
  http_method = aws_api_gateway_method.options_method.http_method
  status_code = aws_api_gateway_method_response.options_response.status_code

  response_parameters = {
    "method.response.header.Access-Control-Allow-Headers" = "'Content-Type,X-Amz-Date,Authorization,X-Api-Key,X-Amz-Security-Token'",
    "method.response.header.Access-Control-Allow-Methods" = "'OPTIONS,POST'",
    "method.response.header.Access-Control-Allow-Origin"  = "'*'"
  }
  depends_on = [aws_api_gateway_integration.options_integration]
}

# --- Deployment Resources ---

resource "aws_lambda_permission" "api_gateway_permission" {
  statement_id  = "AllowAPIGatewayInvoke"
  action        = "lambda:InvokeFunction"
  function_name = aws_lambda_function.image_analyzer_lambda.function_name
  principal     = "apigateway.amazonaws.com"
  source_arn    = "${aws_api_gateway_rest_api.api.execution_arn}/*/*"
}

resource "aws_api_gateway_deployment" "deployment" {
  rest_api_id = aws_api_gateway_rest_api.api.id

  # This ensures a new deployment happens when any part of the API changes.
  triggers = {
    redeployment = sha1(jsonencode(aws_api_gateway_rest_api.api.body))
  }

  lifecycle {
    create_before_destroy = true
  }
}

resource "aws_api_gateway_stage" "stage" {
  deployment_id = aws_api_gateway_deployment.deployment.id
  rest_api_id   = aws_api_gateway_rest_api.api.id
  stage_name    = "v1"
}

# ==============================================================================
# S3 Bucket for Frontend Hosting (Modern Syntax)
# ==============================================================================

resource "random_id" "bucket_suffix" {
  byte_length = 8
}

resource "aws_s3_bucket" "frontend_bucket" {
  bucket = "${var.project_name}-frontend-${random_id.bucket_suffix.hex}"
  force_destroy = true

  tags = {
    Name        = "${var.project_name}-frontend"
    Environment = var.environment
  }
}

resource "aws_s3_bucket_website_configuration" "frontend_website" {
  bucket = aws_s3_bucket.frontend_bucket.id

  index_document {
    suffix = "index.html"
  }

  error_document {
    key = "index.html"
  }

  depends_on = [aws_s3_bucket.frontend_bucket]
}

resource "aws_s3_bucket_public_access_block" "frontend_public_access" {
  bucket                  = aws_s3_bucket.frontend_bucket.id
  block_public_acls       = false
  block_public_policy     = false
  ignore_public_acls      = false
  restrict_public_buckets = false

  depends_on = [aws_s3_bucket.frontend_bucket]
}

resource "aws_s3_bucket_policy" "frontend_policy" {
  bucket = aws_s3_bucket.frontend_bucket.id

  policy = jsonencode({
    Version = "2012-10-17",
    Statement = [
      {
        Effect    = "Allow",
        Principal = "*",
        Action    = "s3:GetObject",
        Resource  = "${aws_s3_bucket.frontend_bucket.arn}/*"
      }
    ]
  })

  depends_on = [aws_s3_bucket_public_access_block.frontend_public_access]
}

# ==============================================================================
# Optional Output
# ==============================================================================
output "frontend_website_url" {
  value = aws_s3_bucket_website_configuration.frontend_website.website_endpoint
}
```
</details>

This file will output the API Gateway URL and the S3 website endpoint after Terraform has finished deploying the resources.

<details>
<summary><code>terraform/outputs.tf</code></summary>

```tf
output "api_gateway_url" {
  description = "The invoke URL of the deployed API"
  value       = aws_api_gateway_stage.stage.invoke_url
}

output "lambda_function_name" {
  description = "The name of the Lambda function"
  value       = aws_lambda_function.image_analyzer_lambda.function_name
}

output "frontend_bucket_name" {
  description = "The name of the S3 bucket hosting the frontend"
  value       = aws_s3_bucket.frontend_bucket.bucket
}

output "frontend_website_endpoint" {
  description = "The website endpoint of the frontend S3 bucket"
  value       = aws_s3_bucket_website_configuration.frontend_website.website_endpoint
}
```
</details>


## ➡️ Step 4 - Frontend Development

Build a stylish chat interface using pure HTML + CSS + JavaScript — no frameworks, easy to deploy via S3 or Amplify.
I have a sample that we'll use for this tutorial, feel free to copy and use it for this demo.

1. Open your code editor (VS Code)
2. Create an `index.html` file 
3. Copy and paste the code below

<details>
<summary><code>index.html</code></summary>

```html
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0"/>
  <title>AI Chatbot using Amazon Bedrock</title>
  <style>
    body {
      margin: 0;
      font-family: 'Segoe UI', sans-serif;
      background-color: #f5f8fa;
      display: flex;
      flex-direction: column;
      height: 100vh;
    }

    header {
      background-color: #232f3e;
      color: white;
      padding: 1rem;
      text-align: center;
      font-size: 1.5rem;
      font-weight: bold;
    }

    #chatBox {
      flex: 1;
      padding: 1rem;
      overflow-y: auto;
      display: flex;
      flex-direction: column;
      gap: 1rem;
    }

    .message {
      max-width: 80%;
      padding: 0.75rem 1rem;
      border-radius: 12px;
      font-size: 1rem;
      line-height: 1.4;
    }

    .user {
      align-self: flex-end;
      background-color: #0073bb;
      color: white;
    }

    .bot {
      align-self: flex-start;
      background-color: #e1ecf4;
      color: #333;
    }

    footer {
      padding: 1rem;
      display: flex;
      gap: 0.5rem;
      background-color: white;
      border-top: 1px solid #ddd;
    }

    input[type="text"] {
      flex: 1;
      padding: 0.75rem;
      font-size: 1rem;
      border: 1px solid #ccc;
      border-radius: 8px;
      outline: none;
    }

    button {
      padding: 0.75rem 1rem;
      background-color: #ff9900;
      border: none;
      border-radius: 8px;
      color: white;
      font-weight: bold;
      cursor: pointer;
      transition: background-color 0.3s ease;
    }

    button:hover {
      background-color: #e48c00;
    }
  </style>
</head>
<body>

  <header>🤖 AI Chatbot — Powered by Amazon Bedrock</header>

  <div id="chatBox"></div>

  <footer>
    <input type="text" id="userInput" placeholder="Type your message..." />
    <button onclick="sendMessage()">Send</button>
  </footer>

  <script>
    let history = [];

    async function sendMessage() {
      const userInput = document.getElementById('userInput');
      const message = userInput.value.trim();
      if (!message) return;

      // Show user message
      addMessage(message, 'user');

      // Call backend
      const response = await fetch('https://your-api-id.execute-api.us-east-1.amazonaws.com/chat', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ message, history })
      });

      const data = await response.json();
      const botReply = data.response;

      // Show bot message
      addMessage(botReply, 'bot');

      // Update history
      history.push({ user: message, assistant: botReply });

      // Reset input
      userInput.value = '';
    }

    function addMessage(text, sender) {
      const msg = document.createElement('div');
      msg.classList.add('message', sender);
      msg.textContent = text;
      document.getElementById('chatBox').appendChild(msg);
      msg.scrollIntoView({ behavior: 'smooth' });
    }
  </script>

</body>
</html>
```
</details>

⚠️Note: Replace `https://your-api-id.execute-api.us-east-1.amazonaws.com/chat` with your real API Gateway endpoint.


## ➡️ Step 6 - Deploy Frontend Chat UI to an S3 Static Website

We'll deploy our fully serverless AI chatbot to S3 for static website hosting.

1. In the AWS Management Console, navigate to Amazon S3, click on "Create Bucket"
2. For General configuration, choose choose General purpose buckets.
3. Enter a unique bucket name, i'll name `myaichatbotdemo`
4. Make sure you disable "Block all public access" to have public access.
5. Keep everything else as default and click "Create bucket"
6. Upload the `index.html` file that you created in step 5
7. Go to "Properties" and scroll down to "Static Website Hosting" and click on "Edit"
8. Under "Static Website Hosting", choose "Enable"
9. Specify index.html as the index document, then click "Save"
10. Go to "Permissions" under Bucket Policy click "Edit"
11. Paste the Bucket Policy below, that grants read-only access to all objects (s3:GetObject) inside a specific S3 bucket.

```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Sid": "PublicReadGetObject",
      "Effect": "Allow",
      "Principal": "*",
      "Action": "s3:GetObject",
      "Resource": "arn:aws:s3:::your-bucket-name/*"
    }
  ]
}
```
⚠️Note: Replace `your-bucket-name` with your actual bucket name, then click "Save"

12. Go back to the S3 Bucket console, choose Objects, then click on `index.html`
13. To visit your fully serverless AI chatbot Live, click on the Object URL.
14. You should see your AI Chatbot with a stylish chat interface running on Amazon S3.

🏆 Now you can ask the AI Chatbot anything and you will have a real-time AI responses.

## 🗑️ Clean Up Resources

When you’re done, clean up your AWS resources to avoid charges.