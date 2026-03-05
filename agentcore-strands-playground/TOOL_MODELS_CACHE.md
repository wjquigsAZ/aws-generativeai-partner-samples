# Tool Models Cache

This document explains how to generate and use the `tool_models.json` cache file for faster agent startup.

## Overview

The `tool_models.json` file caches the list of Bedrock models that support tool use. This avoids querying Bedrock on every agent startup, significantly reducing cold-start times.

## Generating the Cache

### Before Deployment (Recommended)

Run this on your laptop before deploying to AgentCore Runtime:

```bash
# Generate cache for default region (us-west-2)
python generate_tool_models_cache.py

# Generate cache for specific region
python generate_tool_models_cache.py --region us-east-1

# Generate cache to custom location
python generate_tool_models_cache.py --output agentcore_agent/tool_models.json
```

This will:
1. Query Bedrock for all models in the region
2. Test each model for tool support (takes 1-2 minutes)
3. Save results to `agentcore_agent/tool_models.json`

### Verify Existing Cache

```bash
# Check if cache exists and view contents
python generate_tool_models_cache.py --verify

# Verify cache at custom location
python generate_tool_models_cache.py --verify --output path/to/tool_models.json
```

## Deployment Workflow

### Option 1: Pre-generate (Recommended)

```bash
# 1. Generate cache locally
python generate_tool_models_cache.py

# 2. Deploy agent (cache file is included)
cd agentcore_agent
agentcore deploy
```

### Option 2: Auto-generate on First Run

If you don't pre-generate the cache:
- First agent invocation will be slow (queries Bedrock)
- Cache file is created automatically
- Subsequent invocations are fast

## Cache File Format

```json
{
  "region": "us-west-2",
  "models": [
    "us.amazon.nova-micro-v1:0",
    "us.amazon.nova-lite-v1:0",
    "us.amazon.nova-pro-v1:0",
    "anthropic.claude-3-5-sonnet-20241022-v2:0"
  ],
  "generated_at": "2026-03-05T12:34:56Z",
  "generated_by": "generate_tool_models_cache.py"
}
```

## When to Regenerate

Regenerate the cache when:
- Deploying to a new AWS region
- New Bedrock models are released
- Models gain/lose tool support capabilities

## Troubleshooting

### Cache not found during runtime

The agent will automatically query Bedrock and create the cache as a fallback. Check logs for:
```
WARNING: tool_models.json not found. Querying Bedrock as fallback...
```

### Permission errors

Ensure your AWS credentials have:
- `bedrock:ListFoundationModels`
- `bedrock:InvokeModel`

### Empty model list

If no models are found:
1. Check your AWS region has Bedrock enabled
2. Verify model access is granted in Bedrock console
3. Check AWS credentials are valid

## Files

- `generate_tool_models_cache.py` - CLI tool to generate cache
- `agentcore_agent/tool_models.json` - Cache file (gitignored)
- `agentcore_agent/runtime_agent.py` - Runtime agent that uses cache
- `br_utils.py` - Bedrock query utilities
