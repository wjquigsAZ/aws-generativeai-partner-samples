"""
Bedrock Utilities Module

This module provides helper functions for interacting with Amazon Bedrock,
including listing enabled foundation models in the configured AWS region.
"""

import os
import logging
import asyncio
from typing import List, Dict, Optional
import boto3
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Set up logging
logger = logging.getLogger(__name__)
logging.basicConfig(
    format="%(levelname)s | %(name)s | %(message)s",
    level=logging.INFO
)

def get_bedrock_models(region: Optional[str] = None) -> List[str]:
    """
    Synchronous wrapper for get_bedrock_models_async.
    Gets a list of Bedrock models that support tool calling.
    
    Args:
        region: AWS region name. If None, uses AWS_REGION from environment or defaults to us-west-2.
    
    Returns:
        List of model identifier strings that support tool calling.
    """
    return asyncio.run(get_bedrock_models_async(region))

async def test_model_tool_support(
    model: Dict,
    region: str,
    bedrock_runtime,
    test_tool_config: Dict
) -> Optional[str]:
    """
    Test a single model for tool support asynchronously.
    
    Args:
        model: Model dictionary from list_foundation_models
        region: AWS region name
        bedrock_runtime: Bedrock runtime client
        test_tool_config: Tool configuration for testing
    
    Returns:
        Model ID if it supports tools, None otherwise
    """
    model_id = model.get('modelId')
    inference_types = model.get('inferenceTypesSupported', [])
    
    # For INFERENCE_PROFILE models, construct the region-specific ID
    if 'INFERENCE_PROFILE' in inference_types and model_id.startswith('amazon.'):
        region_prefix_map = {
            'us-east-1': 'us',
            'us-west-2': 'us',
            'eu-west-1': 'eu',
            'eu-central-1': 'eu',
            'ap-southeast-1': 'apac',
            'ap-northeast-1': 'apac',
        }
        region_prefix = region_prefix_map.get(region, 'us')
        invocation_model_id = f"{region_prefix}.{model_id}"
        logger.debug(f"Using inference profile ID: {invocation_model_id} for {model_id}")
    else:
        invocation_model_id = model_id
    
    try:
        # Run the synchronous boto3 call in a thread pool
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(
            None,
            lambda: bedrock_runtime.converse(
                modelId=invocation_model_id,
                messages=[{"role": "user", "content": [{"text": "test"}]}],
                toolConfig=test_tool_config,
                inferenceConfig={"maxTokens": 1, "temperature": 0}
            )
        )
        
        logger.info(f"  ✓ {invocation_model_id} - supports tools")
        return invocation_model_id
        
    except bedrock_runtime.exceptions.ValidationException as e:
        error_msg = str(e)
        if 'tool' in error_msg.lower() or 'toolconfig' in error_msg.lower():
            logger.debug(f"  ✗ {model_id} - no tool support: {error_msg}")
        else:
            logger.warning(f"  ? {model_id} - validation error: {error_msg}")
        return None
        
    except Exception as e:
        error_type = type(e).__name__
        logger.warning(f"  ? {model_id} - {error_type}: {str(e)[:100]}")
        return None


async def get_bedrock_models_async(region: Optional[str] = None) -> List[str]:
    """
    Asynchronously get a list of Bedrock foundation models that support tool calling.
    
    This function queries all available models in the specified region and tests
    each one concurrently with a ToolConfig API call. Only models that support 
    tool calling are included in the returned list.
    
    Args:
        region: AWS region name. If None, uses AWS_REGION from environment or defaults to us-west-2.
    
    Returns:
        List of model identifier strings that support tool calling.
    """
    # Get region from parameter, environment, or default
    if region is None:
        region = os.getenv('AWS_REGION', 'us-west-2')
    
    logger.info(f"Querying Bedrock models in region: {region}")
    
    # Create Bedrock clients
    bedrock_client = boto3.client('bedrock', region_name=region)
    bedrock_runtime = boto3.client('bedrock-runtime', region_name=region)
    
    # Get all foundation models
    try:
        response = bedrock_client.list_foundation_models()
        all_models = response.get('modelSummaries', [])
        logger.info(f"Found {len(all_models)} total models in region")
    except Exception as e:
        logger.error(f"Error listing foundation models: {e}")
        return []
    
    # Filter for active models with inference support
    active_models = [
        model for model in all_models
        if model.get('modelLifecycle', {}).get('status') == 'ACTIVE'
        and (
            'ON_DEMAND' in model.get('inferenceTypesSupported', [])
            or 'INFERENCE_PROFILE' in model.get('inferenceTypesSupported', [])
        )
    ]
    logger.info(f"Found {len(active_models)} active models with inference support")
    logger.info("Active models found:")
    for model in active_models:
        logger.info(f"  - {model.get('modelId')}")
    
    # Define a minimal tool config for testing
    test_tool_config = {
        "tools": [
            {
                "toolSpec": {
                    "name": "test_tool",
                    "description": "A test tool",
                    "inputSchema": {
                        "json": {
                            "type": "object",
                            "properties": {
                                "test_param": {
                                    "type": "string",
                                    "description": "A test parameter"
                                }
                            },
                            "required": ["test_param"]
                        }
                    }
                }
            }
        ]
    }
    
    # Test all models concurrently
    logger.info(f"Testing {len(active_models)} models for tool support concurrently...")
    tasks = [
        test_model_tool_support(model, region, bedrock_runtime, test_tool_config)
        for model in active_models
    ]
    
    results = await asyncio.gather(*tasks)
    
    # Filter out None values (models that don't support tools)
    tool_supported_models = [model_id for model_id in results if model_id is not None]
    
    # Sort models: nova-micro first, then Amazon models (sorted), then all others (sorted)
    nova_micro = [m for m in tool_supported_models if 'nova-micro' in m.lower()]
    amazon_models = sorted([m for m in tool_supported_models if 'amazon' in m.lower() and 'nova-micro' not in m.lower()])
    other_models = sorted([m for m in tool_supported_models if 'amazon' not in m.lower()])
    sorted_models = nova_micro + amazon_models + other_models
    
    logger.info(f"Found {len(sorted_models)} models with tool support")
    return sorted_models

if __name__ == "__main__":
    # Example usage when run directly
    import sys
    
    # Check if user wants static list or dynamic query
    if len(sys.argv) > 1 and sys.argv[1] == '--static':
        print("Fetching static list of Bedrock models...")
        models = get_bedrock_models_static()
        print(f"\nStatic Bedrock Models ({len(models)} total):\n")
    else:
        print("Querying Bedrock for models with tool support...")
        print("(This may take a minute as we test each model)\n")
        models = get_bedrock_models()
        print(f"\nBedrock Models with Tool Support ({len(models)} total):\n")
    
    for model_id in models:
        print(f"  - {model_id}")
    
    print("\nUsage:")
    print("  python br_utils.py          # Query models with tool support (dynamic)")
    print("  python br_utils.py --static # Show static list (fast)")