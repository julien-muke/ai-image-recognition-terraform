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
