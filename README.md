# ![aws](https://github.com/julien-muke/Search-Engine-Website-using-AWS/assets/110755734/01cd6124-8014-4baa-a5fe-bd227844d263) AI Image Recognition System with AWS Bedrock, Rekognition & Terraform

<div align="center">

  <br />
    <a href="https://youtu.be/o4fNDCAqyzM" target="_blank">
      <img src="https://github.com/user-attachments/assets/9692b53b-d569-4f6b-a959-39a4dac923a0" alt="Project Banner">
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

<a href="https://youtu.be/o4fNDCAqyzM" target="_blank"><img src="https://github.com/sujatagunale/EasyRead/assets/151519281/1736fca5-a031-4854-8c09-bc110e3bc16d" /></a>

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
 
• **[AWS Account](https://aws.amazon.com/resources/create-account/)**: An active AWS account with administrative privileges to create the necessary resources.<br>
• **[AWS CLI](https://docs.aws.amazon.com/streams/latest/dev/setup-awscli.html)**: The AWS Command Line Interface installed and configured with your credentials.<br>
• **[Terraform](https://developer.hashicorp.com/terraform/install)**: Terraform installed on your local machine. You can verify the installation by running `terraform --version`<br>
• **[Node.js, npm](https://nodejs.org/en/download) and [Python](https://www.python.org/downloads/)**: Required for managing frontend dependencies if you choose to expand the project.<br>
• **Model Access in [Amazon Bedrock](https://aws.amazon.com/bedrock/)**: You must enable access to the foundation models you intend to use. For this project, navigate to the Amazon Bedrock console, go to Model access, and request access to Titan Image Generator G1.<br>

## ➡️ Step 1 - Project Structure

First, let's organize our project files. Create a main directory for your project, and inside it, create the following structure:

    ai-image-recognition-terraform/
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

variable "environment" {
  description = "Deployment environment (e.g., dev, staging, prod)."
  type        = string
  default     = "dev"
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
    redeployment = sha1(jsonencode([
      aws_api_gateway_resource.resource.id,
      aws_api_gateway_method.method.id,
      aws_api_gateway_integration.integration.id,
    ]))
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

Now we'll create the user interface that interacts with our backend.

This is the main HTML file for our application.

<details>
<summary><code>frontend/index.html</code></summary>

```html
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>AI Image Analyzer</title>
    <link rel="stylesheet" href="style.css">
</head>
<body>
    <div class="container">
        <header>
            <h1>AI-Powered Image Analyzer</h1>
            <p>Upload an image to detect labels with AWS Rekognition and get a description from Amazon Bedrock.</p>
        </header>

        <main>
            <div class="upload-area">
                <input type="file" id="imageUpload" accept="image/png, image/jpeg">
                <label for="imageUpload" id="uploadLabel">
                    <span>Click to select an image</span>
                </label>
                <button id="analyzeBtn" disabled>Analyze Image</button>
            </div>

            <div id="preview">
                <img id="imagePreview" src="#" alt="Image Preview">
            </div>

            <div id="results" class="hidden">
                <h2>Analysis Results</h2>
                <div id="loader" class="loader"></div>
                <div id="resultContent">
                     <h3>AI Generated Description:</h3>
                     <p id="description"></p>
                     <h3>Detected Labels:</h3>
                     <div id="labels"></div>
                </div>
            </div>
        </main>

        <footer>
            <p>Built with AWS Rekognition, Bedrock & Terraform</p>
        </footer>
    </div>
    <script src="script.js"></script>
</body>
</html>
```
</details>

Here is some CSS to make the interface look professional and modern.

<details>
<summary><code>frontend/style.css</code></summary>

```css
@import url('https://fonts.googleapis.com/css2?family=Roboto:wght@300;400;700&display=swap');

body {
    font-family: 'Roboto', sans-serif;
    background-color: #f0f2f5;
    color: #333;
    margin: 0;
    padding: 20px;
    display: flex;
    justify-content: center;
    align-items: center;
    min-height: 100vh;
}

.container {
    width: 100%;
    max-width: 800px;
    background-color: #ffffff;
    border-radius: 12px;
    box-shadow: 0 4px 20px rgba(0, 0, 0, 0.1);
    padding: 30px;
    box-sizing: border-box;
}

header {
    text-align: center;
    border-bottom: 1px solid #e0e0e0;
    padding-bottom: 20px;
    margin-bottom: 30px;
}

header h1 {
    color: #1a73e8;
    margin: 0;
}

.upload-area {
    text-align: center;
    margin-bottom: 30px;
}

#imageUpload {
    display: none;
}

#uploadLabel {
    display: block;
    padding: 30px;
    border: 2px dashed #1a73e8;
    border-radius: 8px;
    cursor: pointer;
    background-color: #f8f9fa;
    margin-bottom: 20px;
    transition: background-color 0.3s;
}

#uploadLabel:hover {
    background-color: #e8f0fe;
}

#uploadLabel span {
    font-size: 1.2em;
    font-weight: 500;
}

#analyzeBtn {
    background-color: #1a73e8;
    color: white;
    padding: 12px 25px;
    border: none;
    border-radius: 8px;
    font-size: 1em;
    cursor: pointer;
    transition: background-color 0.3s, box-shadow 0.3s;
    box-shadow: 0 2px 5px rgba(0,0,0,0.1);
}

#analyzeBtn:disabled {
    background-color: #cccccc;
    cursor: not-allowed;
}

#analyzeBtn:not(:disabled):hover {
    background-color: #155ab6;
    box-shadow: 0 4px 10px rgba(0,0,0,0.2);
}

#preview {
    text-align: center;
    margin-bottom: 30px;
}

#imagePreview {
    max-width: 100%;
    max-height: 400px;
    border-radius: 8px;
    display: none;
    box-shadow: 0 4px 15px rgba(0, 0, 0, 0.1);
}

#results {
    background-color: #f8f9fa;
    border-radius: 8px;
    padding: 20px;
}

#results.hidden {
    display: none;
}

#resultContent {
    display: none;
}

#description {
    font-size: 1.1em;
    line-height: 1.6;
    margin-bottom: 20px;
    font-style: italic;
    color: #555;
}

#labels {
    display: flex;
    flex-wrap: wrap;
    gap: 10px;
}

.label-tag {
    background-color: #e8f0fe;
    color: #1a73e8;
    padding: 8px 15px;
    border-radius: 20px;
    font-size: 0.9em;
    font-weight: 500;
}

.loader {
    border: 4px solid #f3f3f3;
    border-top: 4px solid #1a73e8;
    border-radius: 50%;
    width: 40px;
    height: 40px;
    animation: spin 1s linear infinite;
    margin: 20px auto;
}

@keyframes spin {
    0% { transform: rotate(0deg); }
    100% { transform: rotate(360deg); }
}

footer {
    text-align: center;
    margin-top: 30px;
    padding-top: 20px;
    border-top: 1px solid #e0e0e0;
    font-size: 0.9em;
    color: #888;
}
```
</details>

This JavaScript file handles the logic for image preview, converting the image to base64, calling the API, and displaying the results.

<details>
<summary><code>frontend/script.js</code></summary>

```js
document.addEventListener('DOMContentLoaded', () => {
    const imageUpload = document.getElementById('imageUpload');
    const uploadLabel = document.getElementById('uploadLabel');
    const analyzeBtn = document.getElementById('analyzeBtn');
    const imagePreview = document.getElementById('imagePreview');
    const previewContainer = document.getElementById('preview');
    const resultsContainer = document.getElementById('results');
    const loader = document.getElementById('loader');
    const resultContent = document.getElementById('resultContent');
    const descriptionEl = document.getElementById('description');
    const labelsEl = document.getElementById('labels');

    const API_ENDPOINT = 'YOUR_API_GATEWAY_INVOKE_URL'; // <-- IMPORTANT: REPLACE THIS

    let base64Image = null;

    imageUpload.addEventListener('change', (event) => {
        const file = event.target.files[0];
        if (file) {
            // Display image preview
            const reader = new FileReader();
            reader.onload = (e) => {
                imagePreview.src = e.target.result;
                imagePreview.style.display = 'block';
                uploadLabel.querySelector('span').textContent = file.name;
                analyzeBtn.disabled = false;
            };
            reader.readAsDataURL(file);

            // Convert image to base64 for sending to API
            const readerForBase64 = new FileReader();
            readerForBase64.onload = (e) => {
                // Remove the data URL prefix (e.g., "data:image/jpeg;base64,")
                base64Image = e.target.result.split(',')[1];
            };
            readerForBase64.readAsDataURL(file);
        }
    });

    analyzeBtn.addEventListener('click', async () => {
        if (!base64Image || API_ENDPOINT === 'YOUR_API_GATEWAY_INVOKE_URL') {
            alert('Please select an image first or configure the API endpoint in script.js.');
            return;
        }

        // Show loader and results section
        resultsContainer.classList.remove('hidden');
        loader.style.display = 'block';
        resultContent.style.display = 'none';
        descriptionEl.textContent = '';
        labelsEl.innerHTML = '';

        try {
            const response = await fetch(API_ENDPOINT, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ image: base64Image }),
            });

            if (!response.ok) {
                const errorData = await response.json();
                throw new Error(errorData.error || `HTTP error! status: ${response.status}`);
            }

            const data = await response.json();

            // Display results
            descriptionEl.textContent = data.description;
            data.labels.forEach(label => {
                const labelTag = document.createElement('div');
                labelTag.className = 'label-tag';
                labelTag.textContent = label;
                labelsEl.appendChild(labelTag);
            });

        } catch (error) {
            console.error('Error:', error);
            descriptionEl.textContent = `An error occurred: ${error.message}`;
        } finally {
            // Hide loader and show content
            loader.style.display = 'none';
            resultContent.style.display = 'block';
        }
    });
});
```
</details>


## ➡️ Step 5 - Deployment and Testing

Now it's time to bring everything online.

### 1. Deploy the Backend with Terraform

• Navigate to the `terraform` directory in your terminal:

```bash
cd ai-image-recognition-terraform/terraform
```

• Initialize Terraform. This will download the necessary provider plugins.

```bash
terraform init
```

• (Optional) Plan the deployment. This shows you what resources Terraform will create.

```bash
terraform plan
```

• Apply the configuration to create the AWS resources. Type `yes` when prompted.

```bash
terraform apply
```

• After the deployment is complete, Terraform will display the outputs. Copy the `api_gateway_invoke_url`

### 2. Configure and Deploy the Frontend

• Open `frontend/script.js` in your text editor.<br>
⚠️Important: You will need to replace `YOUR_API_GATEWAY_INVOKE_URL` with the actual URL you get from the Terraform output. Make sure to add `/analyze` to the end of the URL you copied.<br>
• Now, upload the frontend files (`index.html`, `style.css`, and the updated `script.js`) to the S3 bucket created by Terraform. You can do this via the AWS Management Console or using the AWS CLI.<br>

➖ Find your bucket name in the S3 console (it will be prefixed with `ai-image-analyzer-frontend-hosting`).<br>
➖ Upload the three files from your `frontend` directory into the bucket.<br>
➖ Ensure the files have public read access. Terraform attempts to set this, but you may need to confirm.<br>

### 3. Test the Application

1. Go to your S3 bucket, choose on index.html then open Object URL in your web browser.
2. You should see the "AI Image Analyzer" interface.
3. Click the upload area, select a JPG or PNG image from your computer.
4. The image preview will appear, and the "Analyze Image" button will be enabled.
5. Click the button. The loader will appear while the backend processes the image.
6. After a few moments, the AI-generated description and the list of detected labels will be displayed.

![Image](https://github.com/user-attachments/assets/f5130fef-d343-40a7-998b-bc065078eb2c)

## 🗑️ Cleaning Up

When you are finished with the project, you can destroy all the created AWS resources to avoid incurring further costs.

1. Navigate back to the terraform directory.
2. Run the destroy command:

```bash
terraform destroy
```