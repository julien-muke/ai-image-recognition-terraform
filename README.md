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


## ➡️ Step 4 - Set Up API Gateway

We're will create a REST API, the REST API provides an HTTP endpoint for your Lambda function. API Gateway routes requests to your Lambda function, and then returns the function's response to clients.

1. In the navigation pane search for API Gateway, choose REST API, click "Build"
2. Choose Create API, enter a name `chatbot-api` click "Create API".
3. Once the REST API is created, click on "Create Resource" 
4. Enter a resource, i'll enter `chat`
5. Make sure you Enable `CORS` (Cross Origin Resource Sharing), which will create an OPTIONS method that allows all origins, all methods, and several common headers.
6. Once the resource is created, click on "Create method"
7. For the method type, choose `POST` 
8. For the integration type choose "Lambda function"
9. Make sure you Enable Lambda proxy integration to send the request to your Lambda function as a structured event.
10. Choose the your regoin `us-east-1` then choose your existing Lambda function that you created earlier.
11. Keep everything as default then click "Create method"
12. Back resources, click on "Deploy API"
13. For the deploy stage, create a new stage, i'll name it `dev` then click on "Deploy"

## Test API Gateway

Deploy your API and test using Postman or curl<br>

⚠️Note: Once it's deployed successfully, we'll have an invoke URL that will be our API endpoint, we're going to call it and then it's going to call the Lambda function to generate the response to us so.

## ➡️ Step 5 - Build the Frontend Chat UI

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