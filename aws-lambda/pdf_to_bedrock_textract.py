"""
Alternative Lambda function using Textract for better PDF text extraction
This version uses AWS Textract instead of PyPDF2 for more accurate text extraction
"""

import json
import boto3
import os
import logging
from typing import Dict, Any
from datetime import datetime
import urllib.parse

logger = logging.getLogger()
logger.setLevel(logging.INFO)

# Initialize AWS clients
s3_client = boto3.client('s3')
textract_client = boto3.client('textract')
bedrock_agent_client = boto3.client('bedrock-agent')

# Environment variables
KNOWLEDGE_BASE_ID = os.environ.get('KNOWLEDGE_BASE_ID')
DATA_SOURCE_ID = os.environ.get('DATA_SOURCE_ID')


def extract_text_with_textract(bucket: str, key: str) -> str:
    """
    Extract text from PDF using AWS Textract.
    
    Args:
        bucket: S3 bucket name
        key: S3 object key
        
    Returns:
        Extracted text as string
    """
    try:
        logger.info(f"Starting Textract analysis for s3://{bucket}/{key}")
        
        # Start asynchronous text detection
        response = textract_client.start_document_text_detection(
            DocumentLocation={
                'S3Object': {
                    'Bucket': bucket,
                    'Name': key
                }
            }
        )
        
        job_id = response['JobId']
        logger.info(f"Textract job started: {job_id}")
        
        # Wait for job completion
        import time
        while True:
            result = textract_client.get_document_text_detection(JobId=job_id)
            status = result['JobStatus']
            
            if status == 'SUCCEEDED':
                break
            elif status == 'FAILED':
                raise Exception("Textract job failed")
            
            logger.info(f"Textract job status: {status}")
            time.sleep(5)
        
        # Extract text from all blocks
        text_blocks = []
        next_token = None
        
        while True:
            if next_token:
                result = textract_client.get_document_text_detection(
                    JobId=job_id,
                    NextToken=next_token
                )
            
            for block in result.get('Blocks', []):
                if block['BlockType'] == 'LINE':
                    text_blocks.append(block['Text'])
            
            next_token = result.get('NextToken')
            if not next_token:
                break
        
        full_text = '\n'.join(text_blocks)
        logger.info(f"Extracted {len(full_text)} characters using Textract")
        
        return full_text
        
    except Exception as e:
        logger.error(f"Error extracting text with Textract: {str(e)}")
        raise


def lambda_handler(event: Dict[str, Any], context: Any) -> Dict[str, Any]:
    """
    Lambda handler using Textract for PDF processing.
    """
    logger.info(f"Received event: {json.dumps(event)}")
    
    if not KNOWLEDGE_BASE_ID or not DATA_SOURCE_ID:
        error_msg = "Missing required environment variables"
        logger.error(error_msg)
        return {'statusCode': 500, 'body': json.dumps({'error': error_msg})}
    
    try:
        for record in event.get('Records', []):
            s3_info = record['s3']
            bucket = s3_info['bucket']['name']
            key = urllib.parse.unquote_plus(s3_info['object']['key'])
            
            if not key.lower().endswith('.pdf'):
                logger.warning(f"Skipping non-PDF file: {key}")
                continue
            
            # Extract text using Textract
            text_content = extract_text_with_textract(bucket, key)
            
            # Store processed text
            text_key = key.replace('.pdf', '.txt')
            s3_client.put_object(
                Bucket=bucket,
                Key=text_key,
                Body=text_content.encode('utf-8'),
                ContentType='text/plain'
            )
            
            # Trigger Bedrock ingestion
            response = bedrock_agent_client.start_ingestion_job(
                knowledgeBaseId=KNOWLEDGE_BASE_ID,
                dataSourceId=DATA_SOURCE_ID,
                description=f"Ingestion job for {key}"
            )
            
            logger.info(f"Started ingestion job: {response['ingestionJob']['ingestionJobId']}")
        
        return {
            'statusCode': 200,
            'body': json.dumps({'message': 'Processing complete'})
        }
        
    except Exception as e:
        logger.error(f"Error: {str(e)}")
        return {
            'statusCode': 500,
            'body': json.dumps({'error': str(e)})
        }
