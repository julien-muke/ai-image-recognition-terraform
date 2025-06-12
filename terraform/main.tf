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
