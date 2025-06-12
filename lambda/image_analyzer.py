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