#!/usr/bin/env python3
"""
CLI tool to generate tool_models.json cache file.

This script queries AWS Bedrock to find all models that support tool use
and saves them to a tool_models.json file. Run this before deploying your
agent to AgentCore Runtime to avoid cold-start delays.

Usage:
    python generate_tool_models_cache.py [--region REGION] [--output PATH]

Examples:
    # Generate cache for default region (us-west-2)
    python generate_tool_models_cache.py

    # Generate cache for specific region
    python generate_tool_models_cache.py --region us-east-1

    # Generate cache to specific output path
    python generate_tool_models_cache.py --output agentcore_agent/tool_models.json
"""

import argparse
import json
import os
import sys
from pathlib import Path
from datetime import datetime
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Import br_utils for querying Bedrock
try:
    from br_utils import get_bedrock_models
except ImportError:
    print("Error: br_utils.py not found. Make sure it's in the same directory.")
    sys.exit(1)


def generate_cache(region: str = None, output_path: str = None) -> bool:
    """
    Generate tool_models.json cache file by querying Bedrock.
    
    Args:
        region: AWS region to query (defaults to AWS_REGION env var or us-west-2)
        output_path: Path where to save the cache file (defaults to agentcore_agent/tool_models.json)
    
    Returns:
        True if successful, False otherwise
    """
    # Determine region
    if region is None:
        region = os.getenv('AWS_REGION', 'us-west-2')
    
    # Determine output path
    if output_path is None:
        output_path = Path(__file__).parent / "tool_models.json"
    else:
        output_path = Path(output_path)
    
    print(f"Querying Bedrock models in region: {region}")
    print("This may take 1-2 minutes as we test each model for tool support...\n")
    
    try:
        # Query Bedrock for models with tool support
        models = get_bedrock_models(region)
        
        if not models:
            print("Warning: No models with tool support found!")
            return False
        
        print(f"\nFound {len(models)} models with tool support:")
        for model_id in models:
            print(f"  - {model_id}")
        
        # Create cache data structure
        cache_data = {
            'region': region,
            'models': models,
            'generated_at': datetime.utcnow().isoformat() + 'Z',
            'generated_by': 'generate_tool_models_cache.py'
        }
        
        # Ensure output directory exists
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Write cache file
        with open(output_path, 'w') as f:
            json.dump(cache_data, f, indent=2)
        
        print(f"\n✓ Successfully saved cache to: {output_path}")
        print(f"  Region: {region}")
        print(f"  Models: {len(models)}")
        print(f"  File size: {output_path.stat().st_size} bytes")
        
        return True
        
    except Exception as e:
        print(f"\n✗ Error generating cache: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return False


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description='Generate tool_models.json cache file for AgentCore deployment',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s
  %(prog)s --region us-east-1
  %(prog)s --output my_agent/tool_models.json
  %(prog)s --region eu-west-1 --output eu_agent/tool_models.json
        """
    )
    
    parser.add_argument(
        '--region',
        type=str,
        help='AWS region to query (default: AWS_REGION env var or us-west-2)'
    )
    
    parser.add_argument(
        '--output',
        type=str,
        help='Output path for tool_models.json (default: agentcore_agent/tool_models.json)'
    )
    
    parser.add_argument(
        '--verify',
        action='store_true',
        help='Verify existing cache file without regenerating'
    )
    
    args = parser.parse_args()
    
    # Handle verify mode
    if args.verify:
        output_path = Path(args.output) if args.output else Path(__file__).parent / "agentcore_agent" / "tool_models.json"
        
        if not output_path.exists():
            print(f"✗ Cache file not found: {output_path}")
            sys.exit(1)
        
        try:
            with open(output_path, 'r') as f:
                data = json.load(f)
            
            print(f"✓ Cache file exists: {output_path}")
            print(f"  Region: {data.get('region', 'unknown')}")
            print(f"  Models: {len(data.get('models', []))}")
            print(f"  Generated: {data.get('generated_at', 'unknown')}")
            print(f"\nModels:")
            for model_id in data.get('models', []):
                print(f"  - {model_id}")
            sys.exit(0)
            
        except Exception as e:
            print(f"✗ Error reading cache file: {e}")
            sys.exit(1)
    
    # Generate cache
    success = generate_cache(region=args.region, output_path=args.output)
    sys.exit(0 if success else 1)


if __name__ == '__main__':
    main()
