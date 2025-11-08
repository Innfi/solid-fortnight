#!/bin/bash

# Deployment script for Lambda function
# This script packages the Lambda function and its dependencies

FUNCTION_NAME="pdf-to-bedrock-kb"
LAYER_NAME="pdf-processing-layer"
PYTHON_VERSION="python3.11"

echo "Building Lambda deployment package..."

# Create deployment directory
mkdir -p deployment
cd deployment

# Install dependencies to package
echo "Installing dependencies..."
mkdir -p python/lib/${PYTHON_VERSION}/site-packages
pip install -r ../requirements.txt -t python/lib/${PYTHON_VERSION}/site-packages

# Create Lambda Layer zip
echo "Creating Lambda Layer..."
zip -r ${LAYER_NAME}.zip python/
echo "Lambda Layer created: ${LAYER_NAME}.zip"

# Create function deployment package
cd ..
mkdir -p function_package
cp pdf_to_bedrock_kb.py function_package/lambda_function.py
cd function_package
zip -r ../deployment/${FUNCTION_NAME}.zip .
cd ..

echo "Lambda function package created: ${FUNCTION_NAME}.zip"

# Clean up
rm -rf function_package

echo ""
echo "Deployment packages created in ./deployment/"
echo ""
echo "Next steps:"
echo "1. Upload ${LAYER_NAME}.zip as a Lambda Layer"
echo "2. Upload ${FUNCTION_NAME}.zip as Lambda function code"
echo "3. Configure environment variables (see environment_variables.json)"
echo "4. Set up S3 trigger for PDF uploads"