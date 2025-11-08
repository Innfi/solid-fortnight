"""
AWS Lambda Function: PDF to Bedrock Knowledge Base
===================================================

This Lambda function reads PDF files from an S3 bucket and ingests them
into an AWS Bedrock Knowledge Base for retrieval-augmented generation (RAG).

Features:
- Triggered by S3 events when PDFs are uploaded
- Extracts text from PDF files
- Ingests content into Bedrock Knowledge Base
- Handles errors and provides detailed logging
- Supports batch processing

Environment Variables Required:
- KNOWLEDGE_BASE_ID: The ID of your Bedrock Knowledge Base
- DATA_SOURCE_ID: The ID of your Bedrock Data Source
- BUCKET_NAME: S3 bucket name (optional, can be from event)
"""

import json
import boto3
import os
import logging
from typing import Dict, Any, List
from datetime import datetime
import PyPDF2
from io import BytesIO
import urllib.parse

# Initialize logger
logger = logging.getLogger()
logger.setLevel(logging.INFO)

# Initialize AWS clients
s3_client = boto3.client('s3')
bedrock_agent_client = boto3.client('bedrock-agent')

# Environment variables
KNOWLEDGE_BASE_ID = os.environ.get('KNOWLEDGE_BASE_ID')
DATA_SOURCE_ID = os.environ.get('DATA_SOURCE_ID')


def extract_text_from_pdf(pdf_bytes: bytes) -> str:
    """
    Extract text content from PDF bytes.
    
    Args:
        pdf_bytes: PDF file content as bytes
        
    Returns:
        Extracted text as string
    """
    try:
        pdf_file = BytesIO(pdf_bytes)
        pdf_reader = PyPDF2.PdfReader(pdf_file)
        
        text_content = []
        for page_num in range(len(pdf_reader.pages)):
            page = pdf_reader.pages[page_num]
            text = page.extract_text()
            if text.strip():
                text_content.append(f"--- Page {page_num + 1} ---\n{text}")
        
        full_text = "\n\n".join(text_content)
        logger.info(f"Extracted {len(full_text)} characters from {len(pdf_reader.pages)} pages")
        
        return full_text
        
    except Exception as e:
        logger.error(f"Error extracting text from PDF: {str(e)}")
        raise


def get_pdf_metadata(pdf_bytes: bytes, s3_key: str) -> Dict[str, Any]:
    """
    Extract metadata from PDF file.
    
    Args:
        pdf_bytes: PDF file content as bytes
        s3_key: S3 object key
        
    Returns:
        Dictionary containing PDF metadata
    """
    try:
        pdf_file = BytesIO(pdf_bytes)
        pdf_reader = PyPDF2.PdfReader(pdf_file)
        
        metadata = {
            'source': s3_key,
            'page_count': len(pdf_reader.pages),
            'processed_date': datetime.utcnow().isoformat(),
        }
        
        # Add PDF metadata if available
        if pdf_reader.metadata:
            if pdf_reader.metadata.title:
                metadata['title'] = pdf_reader.metadata.title
            if pdf_reader.metadata.author:
                metadata['author'] = pdf_reader.metadata.author
            if pdf_reader.metadata.subject:
                metadata['subject'] = pdf_reader.metadata.subject
            if pdf_reader.metadata.creator:
                metadata['creator'] = pdf_reader.metadata.creator
        
        return metadata
        
    except Exception as e:
        logger.warning(f"Error extracting metadata: {str(e)}")
        return {
            'source': s3_key,
            'processed_date': datetime.utcnow().isoformat(),
        }


def read_pdf_from_s3(bucket: str, key: str) -> bytes:
    """
    Read PDF file from S3 bucket.
    
    Args:
        bucket: S3 bucket name
        key: S3 object key
        
    Returns:
        PDF content as bytes
    """
    try:
        logger.info(f"Reading PDF from s3://{bucket}/{key}")
        response = s3_client.get_object(Bucket=bucket, Key=key)
        pdf_bytes = response['Body'].read()
        logger.info(f"Successfully read {len(pdf_bytes)} bytes from S3")
        return pdf_bytes
        
    except Exception as e:
        logger.error(f"Error reading PDF from S3: {str(e)}")
        raise


def ingest_to_bedrock_knowledge_base(
    content: str,
    metadata: Dict[str, Any],
    document_id: str
) -> Dict[str, Any]:
    """
    Ingest document content into Bedrock Knowledge Base.
    
    Args:
        content: Document text content
        metadata: Document metadata
        document_id: Unique document identifier
        
    Returns:
        Response from Bedrock API
    """
    try:
        logger.info(f"Ingesting document {document_id} to Knowledge Base {KNOWLEDGE_BASE_ID}")
        
        # Start ingestion job
        response = bedrock_agent_client.start_ingestion_job(
            knowledgeBaseId=KNOWLEDGE_BASE_ID,
            dataSourceId=DATA_SOURCE_ID,
            description=f"Ingestion job for {metadata.get('source', document_id)}"
        )
        
        ingestion_job_id = response['ingestionJob']['ingestionJobId']
        logger.info(f"Started ingestion job: {ingestion_job_id}")
        
        return {
            'ingestionJobId': ingestion_job_id,
            'status': response['ingestionJob']['status'],
            'document_id': document_id
        }
        
    except Exception as e:
        logger.error(f"Error ingesting to Bedrock Knowledge Base: {str(e)}")
        raise


def store_processed_pdf_to_s3(
    bucket: str,
    content: str,
    metadata: Dict[str, Any],
    original_key: str
) -> str:
    """
    Store processed PDF content as text in S3 (for Bedrock to access).
    This is the recommended approach for Bedrock Knowledge Bases.
    
    Args:
        bucket: S3 bucket name
        content: Extracted text content
        metadata: Document metadata
        original_key: Original PDF key
        
    Returns:
        S3 key of the stored text file
    """
    try:
        # Create a text file key from the PDF key
        text_key = original_key.replace('.pdf', '.txt').replace('.PDF', '.txt')
        if not text_key.endswith('.txt'):
            text_key += '.txt'
        
        # Add metadata as custom headers
        s3_metadata = {
            'original-file': original_key,
            'processed-date': metadata.get('processed_date', ''),
            'page-count': str(metadata.get('page_count', 0))
        }
        
        # Upload text content to S3
        s3_client.put_object(
            Bucket=bucket,
            Key=text_key,
            Body=content.encode('utf-8'),
            ContentType='text/plain',
            Metadata=s3_metadata
        )
        
        logger.info(f"Stored processed content to s3://{bucket}/{text_key}")
        return text_key
        
    except Exception as e:
        logger.error(f"Error storing processed content to S3: {str(e)}")
        raise


def process_pdf_event(event_record: Dict[str, Any]) -> Dict[str, Any]:
    """
    Process a single S3 event record containing PDF upload information.
    
    Args:
        event_record: S3 event record
        
    Returns:
        Processing result
    """
    try:
        # Extract S3 information from event
        s3_info = event_record['s3']
        bucket = s3_info['bucket']['name']
        key = urllib.parse.unquote_plus(s3_info['object']['key'])
        
        logger.info(f"Processing PDF: {key} from bucket: {bucket}")
        
        # Validate that it's a PDF file
        if not key.lower().endswith('.pdf'):
            logger.warning(f"Skipping non-PDF file: {key}")
            return {
                'status': 'skipped',
                'reason': 'Not a PDF file',
                'key': key
            }
        
        # Read PDF from S3
        pdf_bytes = read_pdf_from_s3(bucket, key)
        
        # Extract text and metadata
        text_content = extract_text_from_pdf(pdf_bytes)
        metadata = get_pdf_metadata(pdf_bytes, key)
        
        # Create unique document ID
        document_id = f"{bucket}/{key}".replace('/', '_')
        
        # Store processed content back to S3 as text
        # Bedrock Knowledge Base can then sync from this text file
        text_key = store_processed_pdf_to_s3(bucket, text_content, metadata, key)
        
        # Trigger Bedrock Knowledge Base ingestion
        ingestion_result = ingest_to_bedrock_knowledge_base(
            content=text_content,
            metadata=metadata,
            document_id=document_id
        )
        
        return {
            'status': 'success',
            'bucket': bucket,
            'original_key': key,
            'processed_key': text_key,
            'document_id': document_id,
            'metadata': metadata,
            'ingestion_result': ingestion_result
        }
        
    except Exception as e:
        logger.error(f"Error processing PDF event: {str(e)}")
        return {
            'status': 'failed',
            'error': str(e),
            'key': key if 'key' in locals() else 'unknown'
        }


def lambda_handler(event: Dict[str, Any], context: Any) -> Dict[str, Any]:
    """
    Main Lambda handler function.
    
    Args:
        event: Lambda event (S3 event)
        context: Lambda context
        
    Returns:
        Processing results
    """
    logger.info(f"Received event: {json.dumps(event)}")
    
    # Validate environment variables
    if not KNOWLEDGE_BASE_ID or not DATA_SOURCE_ID:
        error_msg = "Missing required environment variables: KNOWLEDGE_BASE_ID or DATA_SOURCE_ID"
        logger.error(error_msg)
        return {
            'statusCode': 500,
            'body': json.dumps({'error': error_msg})
        }
    
    results = []
    
    try:
        # Process each S3 event record
        if 'Records' in event:
            for record in event['Records']:
                result = process_pdf_event(record)
                results.append(result)
        else:
            # Handle direct invocation (for testing)
            logger.info("Direct invocation detected")
            if 'bucket' in event and 'key' in event:
                # Simulate S3 event structure
                simulated_record = {
                    's3': {
                        'bucket': {'name': event['bucket']},
                        'object': {'key': event['key']}
                    }
                }
                result = process_pdf_event(simulated_record)
                results.append(result)
            else:
                error_msg = "Invalid event format. Expected S3 event or {bucket, key}"
                logger.error(error_msg)
                return {
                    'statusCode': 400,
                    'body': json.dumps({'error': error_msg})
                }
        
        # Summarize results
        success_count = sum(1 for r in results if r.get('status') == 'success')
        failed_count = sum(1 for r in results if r.get('status') == 'failed')
        skipped_count = sum(1 for r in results if r.get('status') == 'skipped')
        
        logger.info(f"Processing complete. Success: {success_count}, Failed: {failed_count}, Skipped: {skipped_count}")
        
        return {
            'statusCode': 200,
            'body': json.dumps({
                'message': 'Processing complete',
                'summary': {
                    'total': len(results),
                    'success': success_count,
                    'failed': failed_count,
                    'skipped': skipped_count
                },
                'results': results
            })
        }
        
    except Exception as e:
        logger.error(f"Unhandled error in lambda_handler: {str(e)}")
        return {
            'statusCode': 500,
            'body': json.dumps({
                'error': 'Internal server error',
                'message': str(e)
            })
        }
