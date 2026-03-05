"""
Bedrock Utilities Module

This module provides helper functions for interacting with Amazon Bedrock,
including loading cached tool-supporting models from tool_models.json.
"""

import json
import logging
import sys
from pathlib import Path
from typing import List

# Set up logging
logger = logging.getLogger(__name__)
logging.basicConfig(
    format="%(levelname)s | %(name)s | %(message)s",
    level=logging.INFO
)

# Path to the cached tool models file
TOOL_MODELS_FILE = Path(__file__).parent / "tool_models.json"


def get_bedrock_models(region: str = None) -> List[str]:
    """
    Get a list of Bedrock models that support tool calling from tool_models.json.
    
    Args:
        region: Unused, kept for backward compatibility.
    
    Returns:
        List of model identifier strings that support tool calling.
    
    Raises:
        SystemExit: If tool_models.json is not found.
    """
    if not TOOL_MODELS_FILE.exists():
        logger.error(
            f"tool_models.json not found at {TOOL_MODELS_FILE}. "
            "Run 'python generate_tool_models_cache.py' to generate it."
        )
        sys.exit(1)

    try:
        with open(TOOL_MODELS_FILE, 'r') as f:
            data = json.load(f)
        
        models = data.get('models', [])
        cached_region = data.get('region', 'unknown')
        logger.info(f"Loaded {len(models)} tool-supporting models from {TOOL_MODELS_FILE} (region: {cached_region})")
        return models

    except json.JSONDecodeError as e:
        logger.error(f"Invalid JSON in {TOOL_MODELS_FILE}: {e}")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Error reading {TOOL_MODELS_FILE}: {e}")
        sys.exit(1)


if __name__ == "__main__":
    models = get_bedrock_models()
    print(f"\nBedrock Models with Tool Support ({len(models)} total):\n")
    for model_id in models:
        print(f"  - {model_id}")
