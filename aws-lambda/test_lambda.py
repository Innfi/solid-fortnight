"""
Test script for PDF to Bedrock Knowledge Base Lambda function
"""

import json
import sys
sys.path.insert(0, '.')

# Mock AWS clients for local testing
import unittest
from unittest.mock import Mock, patch, MagicMock
from io import BytesIO

# Import the lambda function
import pdf_to_bedrock_kb


class TestPDFToBedrockKB(unittest.TestCase):
    
    def setUp(self):
        """Set up test fixtures"""
        self.sample_event = {
            "Records": [
                {
                    "s3": {
                        "bucket": {
                            "name": "test-bucket"
                        },
                        "object": {
                            "key": "documents/sample.pdf"
                        }
                    }
                }
            ]
        }
        
        # Mock environment variables
        self.env_patcher = patch.dict('os.environ', {
            'KNOWLEDGE_BASE_ID': 'test-kb-id',
            'DATA_SOURCE_ID': 'test-ds-id'
        })
        self.env_patcher.start()
    
    def tearDown(self):
        """Clean up after tests"""
        self.env_patcher.stop()
    
    @patch('pdf_to_bedrock_kb.s3_client')
    @patch('pdf_to_bedrock_kb.bedrock_agent_client')
    def test_process_pdf_event_success(self, mock_bedrock, mock_s3):
        """Test successful PDF processing"""
        
        # Mock S3 response with sample PDF content
        mock_response = MagicMock()
        mock_response.read.return_value = self._create_sample_pdf()
        mock_s3.get_object.return_value = {'Body': mock_response}
        
        # Mock Bedrock response
        mock_bedrock.start_ingestion_job.return_value = {
            'ingestionJob': {
                'ingestionJobId': 'test-job-id',
                'status': 'STARTED'
            }
        }
        
        # Process event record
        result = pdf_to_bedrock_kb.process_pdf_event(self.sample_event['Records'][0])
        
        # Assertions
        self.assertEqual(result['status'], 'success')
        self.assertEqual(result['bucket'], 'test-bucket')
        self.assertEqual(result['original_key'], 'documents/sample.pdf')
        
    @patch('pdf_to_bedrock_kb.s3_client')
    def test_skip_non_pdf_files(self, mock_s3):
        """Test that non-PDF files are skipped"""
        
        event = {
            "s3": {
                "bucket": {"name": "test-bucket"},
                "object": {"key": "documents/sample.txt"}
            }
        }
        
        result = pdf_to_bedrock_kb.process_pdf_event(event)
        
        self.assertEqual(result['status'], 'skipped')
        self.assertEqual(result['reason'], 'Not a PDF file')
    
    @patch('pdf_to_bedrock_kb.s3_client')
    @patch('pdf_to_bedrock_kb.bedrock_agent_client')
    def test_lambda_handler(self, mock_bedrock, mock_s3):
        """Test the main Lambda handler"""
        
        # Mock S3 response
        mock_response = MagicMock()
        mock_response.read.return_value = self._create_sample_pdf()
        mock_s3.get_object.return_value = {'Body': mock_response}
        
        # Mock Bedrock response
        mock_bedrock.start_ingestion_job.return_value = {
            'ingestionJob': {
                'ingestionJobId': 'test-job-id',
                'status': 'STARTED'
            }
        }
        
        # Invoke handler
        context = {}
        result = pdf_to_bedrock_kb.lambda_handler(self.sample_event, context)
        
        # Assertions
        self.assertEqual(result['statusCode'], 200)
        body = json.loads(result['body'])
        self.assertEqual(body['summary']['total'], 1)
    
    def test_direct_invocation(self):
        """Test direct invocation with bucket and key"""
        
        event = {
            "bucket": "test-bucket",
            "key": "sample.pdf"
        }
        
        with patch('pdf_to_bedrock_kb.s3_client') as mock_s3, \
             patch('pdf_to_bedrock_kb.bedrock_agent_client') as mock_bedrock:
            
            # Mock responses
            mock_response = MagicMock()
            mock_response.read.return_value = self._create_sample_pdf()
            mock_s3.get_object.return_value = {'Body': mock_response}
            
            mock_bedrock.start_ingestion_job.return_value = {
                'ingestionJob': {
                    'ingestionJobId': 'test-job-id',
                    'status': 'STARTED'
                }
            }
            
            context = {}
            result = pdf_to_bedrock_kb.lambda_handler(event, context)
            
            self.assertEqual(result['statusCode'], 200)
    
    def _create_sample_pdf(self) -> bytes:
        """Create a simple PDF for testing"""
        # This is a minimal PDF structure
        pdf_content = b"""%PDF-1.4
1 0 obj
<<
/Type /Catalog
/Pages 2 0 R
>>
endobj
2 0 obj
<<
/Type /Pages
/Kids [3 0 R]
/Count 1
>>
endobj
3 0 obj
<<
/Type /Page
/Parent 2 0 R
/Resources <<
/Font <<
/F1 <<
/Type /Font
/Subtype /Type1
/BaseFont /Helvetica
>>
>>
>>
/MediaBox [0 0 612 792]
/Contents 4 0 R
>>
endobj
4 0 obj
<<
/Length 44
>>
stream
BT
/F1 12 Tf
100 700 Td
(Test PDF) Tj
ET
endstream
endobj
xref
0 5
0000000000 65535 f 
0000000009 00000 n 
0000000058 00000 n 
0000000115 00000 n 
0000000317 00000 n 
trailer
<<
/Size 5
/Root 1 0 R
>>
startxref
410
%%EOF"""
        return pdf_content


def run_local_test():
    """Run a simple local test"""
    print("Running local test of PDF to Bedrock KB Lambda function...")
    
    # Create test event
    test_event = {
        "bucket": "my-test-bucket",
        "key": "documents/test.pdf"
    }
    
    print(f"Test event: {json.dumps(test_event, indent=2)}")
    
    # Note: This will fail without actual AWS credentials and resources
    # Use the unittest tests above for mocked testing
    print("\nFor full testing, run: python -m pytest test_lambda.py")
    print("For mocked testing, run: python -m unittest test_lambda.TestPDFToBedrockKB")


if __name__ == "__main__":
    # Run unit tests
    unittest.main(argv=[''], exit=False, verbosity=2)
    
    print("\n" + "="*60)
    run_local_test()
