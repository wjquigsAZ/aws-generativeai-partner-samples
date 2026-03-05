#!/usr/bin/env python3
"""
Runtime-specific agent for AgentCore deployment of Strands Playground.
"""

import json
import os
import logging
import sys
import boto3
from typing import Dict, List
from dotenv import load_dotenv
from duckduckgo_search import DDGS
from bedrock_agentcore.runtime import BedrockAgentCoreApp
from strands import Agent, tool
from strands.models import BedrockModel
from strands.telemetry import StrandsTelemetry
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.sdk.resources import Resource
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from openinference.instrumentation.strands_agents import StrandsAgentsToOpenInferenceProcessor

#Arize Project Name
os.environ["ARIZE_PROJECT_NAME"] = "agentcore-strands-playground" ## <-- Update with your Arize Project Name

strands_processor = StrandsAgentsToOpenInferenceProcessor()
resource = Resource.create({
  "model_id": os.environ["ARIZE_PROJECT_NAME"], 
})
provider = TracerProvider(resource=resource)
provider.add_span_processor(strands_processor)
otel_exporter = OTLPSpanExporter()
provider.add_span_processor(BatchSpanProcessor(otel_exporter))
trace.set_tracer_provider(provider)

# Load environment variables
load_dotenv()

# Import Strands components
from strands import Agent, tool
from strands.models import BedrockModel
from strands_tools import (
    agent_graph, calculator, cron, current_time, editor, environment,
    file_read, file_write, generate_image, http_request, image_reader, journal,
    load_tool, nova_reels,
    #python_repl, 
    retrieve, shell, 
    #slack,
    speak, stop, swarm, think, use_aws, use_llm, workflow
)

# Import BedrockAgentCore runtime
from bedrock_agentcore.runtime import BedrockAgentCoreApp

# Import memory management modules
from bedrock_agentcore.memory.integrations.strands.session_manager import AgentCoreMemorySessionManager
from bedrock_agentcore.memory.integrations.strands.config import AgentCoreMemoryConfig, RetrievalConfig

# Set up logging
logger = logging.getLogger(__name__)
logging.basicConfig(format="%(levelname)s | %(name)s | %(message)s")
logger.setLevel(logging.DEBUG)

# Create BedrockAgentCore app instance
app = BedrockAgentCoreApp()

# Environment variables for configuration
# ordinarily these will all be passed as part of the request payload since they can change per invocation,
# and since they are set by the front-end

SYSTEM_PROMPT = os.getenv('STRANDS_SYSTEM_PROMPT', 
    'You are a helpful assistant powered by Strands. Strands Agents is a simple-to-use, code-first framework for building agents - open source by AWS.')
MODEL_ID = os.getenv('STRANDS_MODEL_ID', 'us.amazon.nova-pro-v1:0')
REGION = os.getenv('AWS_REGION')
if not REGION:
    logger.info("no region configured in env")
    REGION='us-west-2'
MAX_TOKENS = int(os.getenv('STRANDS_MAX_TOKENS', '1000'))
TEMPERATURE = float(os.getenv('STRANDS_TEMPERATURE', '0.3'))
TOP_P = float(os.getenv('STRANDS_TOP_P', '0.9'))
# apparently this env variable is set automatically?
MEMORY_ID = os.getenv('BEDROCK_AGENTCORE_MEMORY_ID')
logger.debug(f"Configured MEMORY_ID: {MEMORY_ID}")
MEMORY_NAMESPACE = os.getenv('BEDROCK_AGENTCORE_MEMORY_NAME')
# Authentication is disabled by default in this playground
ENABLE_AUTH = False

# define a sample tool for web search
@tool
def websearch(
    keywords: str, region: str = "us-en", max_results: int | None = None
) -> str:
    """Search the web to get updated information.
    Args:
        keywords (str): The search query keywords.
        region (str): The search region: wt-wt, us-en, uk-en, ru-ru, etc..
        max_results (int | None): The maximum number of results to return.
    Returns:
        List of dictionaries with search results.
    """
    try:
        # Use duckduckgo-search library
        with DDGS() as ddgs:
            results = list(ddgs.text(keywords, region=region, max_results=max_results or 5))
        
        if results:
            formatted_results = []
            for result in results:
                title = result.get('title', 'No Title')
                body = result.get('body', 'No Description')
                url = result.get('href', 'No URL')  # Note: 'href' instead of 'url'
                formatted_results.append(f"Title: {title}\nDescription: {body}\nURL: {url}\n")
            logger.debug(f"Web search results for '{keywords}': {formatted_results}")
            return "\n".join(formatted_results)
        else:
            return "No results found."
    except Exception as e:
        logger.error(f"Web search error: {e}")
        return f"Search error: {str(e)}"



# Define all available built-in Strands tools
# note that many of these tools (such as use_aws) require IAM permissions
# for the AGENT execution role
available_tools = {
    'agent_graph': agent_graph,
    'calculator': calculator,
    'cron': cron,
    'current_time': current_time, 
    'editor': editor,
    'environment': environment,
    'file_read': file_read,
    'file_write': file_write,
    'generate_image': generate_image,
    'http_request': http_request,
    'image_reader': image_reader,
    'journal': journal,
    'load_tool': load_tool,
    'nova_reels': nova_reels,
#    'python_repl': python_repl, 
    'retrieve': retrieve,
    'shell': shell,
#    'slack': slack, 
    'speak': speak,
    'stop': stop,
    'swarm': swarm,
    'think': think,
    'use_aws': use_aws,
    'use_llm': use_llm,
    'workflow': workflow,
    'websearch': websearch
}

# Agent cache: stores agents by (model_id, session_id, actor_id) to avoid session manager conflicts
# previously cached agents by model_id, and shared same agent for different sessions/actors
# limitation in AC Memory session manager doesn't allow sharing
# "Currently, only one agent per session is supported when using AgentCoreMemorySessionManager."
agent_cache: Dict[tuple[str, str, str], Agent] = {}

# Track first invocation per model
first_invocation_per_model: Dict[str, bool] = {}

# Track current tools per agent for comparison
current_tools_per_agent: Dict[tuple[str, str, str], List[str]] = {}

# Track memory configs per (model_id, session_id, actor_id)
memory_cache: Dict[tuple[str, str, str], AgentCoreMemoryConfig] = {}

# Track session managers per (model_id, session_id, actor_id)
session_manager_cache: Dict[tuple[str, str, str], AgentCoreMemorySessionManager] = {}

def get_tools_from_names(tool_names: List[str]) -> List:
    """
    Convert tool names to actual tool objects.
    """
    tools = []
    for tool_name in tool_names:
        if tool_name in available_tools:
            tools.append(available_tools[tool_name])
        else:
            logger.warning(f"Tool '{tool_name}' not found in available_tools, skipping")
    
    # Fallback to default tools if no valid tools were found
    if not tools:
        logger.warning("No valid tools found, using default set")
        tools = [calculator, current_time, http_request, websearch]
    
    return tools


         
# connect to memory provisioned by agentcore starter toolkit
memory_id = MEMORY_ID

def get_memory_config(model_id: str, session_id: str, actor_id: str, memory_id: str = None) -> AgentCoreMemoryConfig:
    """Get memory config from cache or create new one.
    
    Note: Includes model_id in cache key so each model gets its own memory config.
    This allows switching models within the same session without conflicts.
    """
    # Use provided memory_id, fallback to global MEMORY_ID, or None
    effective_memory_id = memory_id or MEMORY_ID
    logger.debug(f"Retrieving memory config for model: {model_id}, session: {session_id}, actor: {actor_id}, memory_id: {effective_memory_id}")
    cache_key = (model_id, session_id, actor_id)
    
    if cache_key not in memory_cache:
        # Create new memory config
        memory_config = AgentCoreMemoryConfig(
            memory_id=effective_memory_id,
            session_id=session_id,
            actor_id=actor_id,
            retrieval_config={
                f"/users/{actor_id}/facts": RetrievalConfig(top_k=3, relevance_score=0.5),
                f"/users/{actor_id}/preferences": RetrievalConfig(top_k=3, relevance_score=0.5),
                f"/summaries/{actor_id}/{session_id}": RetrievalConfig(top_k=3, relevance_score=0.5),
            }
        )
        memory_cache[cache_key] = memory_config
        logger.debug(f"Created new memory config for model: {model_id}, session: {session_id}, actor: {actor_id}, memory_id: {effective_memory_id}")
    
    return memory_cache[cache_key]

def get_or_create_agent(model_id: str, tool_names: List[str] = None, actor_id: str = None, session_id: str = None, memory_id: str = None) -> Agent:
    """
    Get existing agent from cache or create new one for the given (model_id, session_id, actor_id).
    Each session gets its own agent instance to avoid session manager conflicts.
    Uses Strands' dynamic tool reloading to update tools without creating a new agent.
    
    Args:
        model_id: The Bedrock model identifier
        tool_names: List of tool names to include (defaults to standard set)
        actor_id: The actor/user identifier
        session_id: The session identifier
        memory_id: The memory identifier (optional, uses global MEMORY_ID if not provided)
        
    Returns:
        RuntimeStrandsAgent instance for the specified model and session
    """
    # Create cache key from model_id, session_id, and actor_id
    cache_key = (model_id, session_id, actor_id)
    
    # Check if agent exists for this combination
    if cache_key not in agent_cache:
        logger.info(f"Creating new agent for model: {model_id}")
        
        # Create new Bedrock model instance
        bedrock_model = BedrockModel(
            model_id=model_id,
            region_name=REGION,
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
            top_p=TOP_P,
        )
        
        # Get tool objects from names
        tools = get_tools_from_names(tool_names)
        
        # Use default tool set if none provided
        if tools is None or len(tools) == 0:
            tools = [calculator, current_time]
        
        logger.debug(f"creating agent with model: {model_id}, actor_id: {actor_id}, session_id: {session_id}, memory_id: {memory_id}, total tools: {len(tools)}")

        memory_config = get_memory_config(model_id, session_id, actor_id, memory_id)
        
        # Create session manager and cache it by (model_id, session_id, actor_id)
        # This allows each model to have its own session manager, avoiding conflicts
        # when switching models within the same session
        sm_cache_key = (model_id, session_id, actor_id)
        if sm_cache_key not in session_manager_cache:
            session_manager = AgentCoreMemorySessionManager(memory_config, REGION)
            session_manager_cache[sm_cache_key] = session_manager
            logger.debug(f"Created new session manager for model: {model_id}")
        else:
            session_manager = session_manager_cache[sm_cache_key]
            logger.debug(f"Reusing session manager for model: {model_id}")
        
        # Create and cache new agent with specified tools
        agent_cache[cache_key] = Agent(
            name='Strands_Playground',
            model=bedrock_model,
            system_prompt=SYSTEM_PROMPT,
            tools=tools,
            session_manager=session_manager
        )
        
        # Set state values using the state manager
        agent_cache[cache_key].state.set("actor_id", actor_id)
        agent_cache[cache_key].state.set("session_id", session_id)
        
        # Track tools and first invocation for this model
        current_tools_per_agent[cache_key] = tool_names
        first_invocation_per_model[model_id] = True
        
        logger.info(f"Agent created and cached for model: {model_id}, session: {session_id}, actor: {actor_id} with {len(tools)} tools")
    else:
        logger.info(f"Found cached agent for model: {model_id}, session: {session_id}, actor: {actor_id}")
        logger.debug(f"cache key: {cache_key} agent cache: {agent_cache}")

        # Agent exists - check if tools have changed
        current_tools = current_tools_per_agent.get(cache_key, [])
        
        if set(tool_names) != set(current_tools):
            logger.info(f"Tools changed for agent, dynamically reloading tools")
            logger.debug(f"Previous tools: {current_tools}")
            logger.debug(f"New tools: {tool_names}")
            
            # Get new tool objects
            tools = get_tools_from_names(tool_names)
            if tools and hasattr(agent_cache[cache_key], 'tool_registry'):
                try:
                    processed_tool_names = agent_cache[cache_key].tool_registry.process_tools(tools)
                    logger.info(f"Processed tools: {processed_tool_names}")
                except Exception as e:
                    logger.error(f"Failed to process tools: {e}")
            else:
                logger.error(f"no tools found {tools} in agent: {agent_cache[cache_key]}")

            # Dynamically reload tools on existing agent
            agent_cache[cache_key].tools = tools
            
            # Update tracked tool names (strings)
            current_tools_per_agent[cache_key] = processed_tool_names
            
            logger.info(f"Tools dynamically reloaded: {len(tools)} tools now active")
        else:
            logger.debug(f"Using cached agent with same tools")
    
    return agent_cache[cache_key]

@app.entrypoint
def invoke_agent(payload, **kwargs) -> str:
    logger.debug('invoked runtime_agent')
    """
    Main entrypoint for AgentCore Runtime.
    This function will be called when the agent receives a request.
    Supports dynamic model switching via model_id and dynamic tool reloading via tools in payload.
    """
    try:
        logger.debug(f"Payload received: {json.dumps(payload)}")
        logger.debug(f"Kwargs received: {kwargs.keys()}")
        
        # Extract event from kwargs to access headers
        event = kwargs.get('event', {})
        
        # Extract input, username, model_id, and tools from payload - handle different input formats
        username = None
        model_id = MODEL_ID  # Default model
        tool_names = None  # Will use default if not specified
        memory_id = None
        session_id = None  # Initialize session_id
        
        if isinstance(payload, dict):
            user_input = payload.get('inputText', payload.get('prompt', payload.get('message', str(payload))))
            username = payload.get('username')
            # for now, we're not doing anything with sessionId...it will automatically map to a new path in memory if it changes
            session_id = payload.get('sessionId')
            model_id = payload.get('model_id', MODEL_ID)  # Extract model_id from payload
            memory_id = payload.get('memory_id')
            tool_names = payload.get('tools')  # Extract tools list from payload
            memory_debug = payload.get('memoryDebug', False)  # Extract memoryDebug flag
        elif isinstance(payload, str):
            user_input = payload
            session_id = "default_session"  # Provide default session_id for string payloads
            memory_debug = False
        else:
            user_input = str(payload)
            session_id = "default_session"  # Provide default session_id
            memory_debug = False

        logger.debug(f"Using memory: {memory_id}, debugging {memory_debug}")
        
        # Validate input
        if not user_input or not isinstance(user_input, str):
            raise ValueError(f"Invalid input format: expected string, got {type(user_input)}: {user_input}")
        
        user_input = user_input.strip()
        if not user_input:
            raise ValueError("Empty input provided")
        
        # Log the request details
        num_tools = f" with {len(tool_names)} tools" if tool_names else " with default tools"
        logger.info(f"Received input from user '{username}', session {session_id} using model '{model_id}'{num_tools}: '{user_input}'")
            
        # Get or create agent for the specified model
        agent = get_or_create_agent(model_id, tool_names, username, session_id, memory_id)

        # Retrieve and log memories from session manager (only if memoryDebug is enabled)
        if memory_debug and hasattr(agent, 'session_manager') and isinstance(agent.session_manager, AgentCoreMemorySessionManager):
            try:
                # Get the memory client from session manager
                memory_client = agent.session_manager.memory_client
                memory_config = agent.session_manager.memory_config
                
                logger.info(f"=== Memory Retrieval for session: {session_id}, actor: {username} ===")
                
                # Retrieve memories from different namespaces
                for namespace, retrieval_config in memory_config.retrieval_config.items():
                    try:
                        # Query memories from this namespace
                        response = memory_client.query_memory(
                            memoryId=memory_config.memory_id,
                            text=user_input,  # Use current input as query
                            namespace=namespace,
                            maxResults=retrieval_config.top_k
                        )
                        
                        memories = response.get('memories', [])
                        logger.info(f"Namespace '{namespace}': Found {len(memories)} memories")
                        
                        for idx, memory in enumerate(memories, 1):
                            memory_content = memory.get('content', {}).get('text', 'N/A')
                            relevance_score = memory.get('relevanceScore', 0)
                            logger.debug(f"  Memory {idx} (score: {relevance_score:.3f}): {memory_content[:100]}...")
                            
                    except Exception as ns_error:
                        logger.debug(f"Could not retrieve memories from namespace '{namespace}': {ns_error}")
                
                logger.info("=== End Memory Retrieval ===")
                
            except Exception as mem_error:
                logger.debug(f"Could not retrieve memories: {mem_error}")

        # ===== DIAGNOSTIC LOGGING: Inspect conversation state before invocation =====
        logger.info("=== PRE-INVOCATION CONVERSATION STATE DIAGNOSTIC ===")
        
        try:
            # Access the session manager's conversation history
            if hasattr(agent, 'session_manager') and hasattr(agent.session_manager, 'conversation'):
                conversation = agent.session_manager.conversation
                logger.info(f"Conversation object type: {type(conversation)}")
                logger.info(f"Conversation length: {len(conversation) if hasattr(conversation, '__len__') else 'N/A'}")
                
                # Log full conversation structure
                if hasattr(conversation, 'messages'):
                    messages = conversation.messages
                    logger.info(f"Total messages in conversation: {len(messages)}")
                    
                    # Dump full conversation history for debugging
                    logger.info("=== FULL CONVERSATION HISTORY ===")
                    for idx, msg in enumerate(messages):
                        logger.info(f"Message {idx}:")
                        logger.info(f"  Role: {msg.get('role', 'UNKNOWN')}")
                        
                        content = msg.get('content', [])
                        if isinstance(content, list):
                            logger.info(f"  Content blocks: {len(content)}")
                            for block_idx, block in enumerate(content):
                                if isinstance(block, dict):
                                    block_type = list(block.keys())[0] if block else 'empty'
                                    logger.info(f"    Block {block_idx} type: {block_type}")
                                    
                                    # Log toolUse blocks in detail
                                    if 'toolUse' in block:
                                        tool_use = block['toolUse']
                                        logger.info(f"      toolUseId: {tool_use.get('toolUseId', 'N/A')}")
                                        logger.info(f"      name: {tool_use.get('name', 'N/A')}")
                                    
                                    # Log toolResult blocks in detail
                                    elif 'toolResult' in block:
                                        tool_result = block['toolResult']
                                        logger.info(f"      toolUseId: {tool_result.get('toolUseId', 'N/A')}")
                                        logger.info(f"      status: {tool_result.get('status', 'N/A')}")
                                    
                                    # Log text blocks
                                    elif 'text' in block:
                                        text_preview = block['text'][:100] if len(block['text']) > 100 else block['text']
                                        logger.info(f"      text: {text_preview}...")
                                else:
                                    logger.info(f"    Block {block_idx}: {type(block)}")
                        else:
                            logger.info(f"  Content: {str(content)[:200]}")
                    
                    logger.info("=== END CONVERSATION HISTORY ===")
                    
                    # Validate conversation state
                    logger.info("=== CONVERSATION VALIDATION ===")
                    
                    # Check 1: Conversation must start with user message
                    if messages and len(messages) > 0:
                        first_message = messages[0]
                        first_role = first_message.get('role', 'UNKNOWN')
                        logger.info(f"First message role: {first_role}")
                        
                        if first_role != 'user':
                            logger.warning(f"INVALID STATE: Conversation does not start with 'user' message (starts with '{first_role}')")
                            logger.warning("This will cause 'A conversation must start with a user message' error")
                            logger.warning("Clearing conversation to reset state...")
                            
                            # Clear the conversation
                            conversation.messages = []
                            logger.info("Conversation cleared successfully")
                        else:
                            logger.info("✓ Conversation starts with user message")
                    
                    # Check 2: Look for orphaned toolUse blocks (toolUse without matching toolResult)
                    tool_use_ids = set()
                    tool_result_ids = set()
                    
                    for msg in messages:
                        content = msg.get('content', [])
                        if isinstance(content, list):
                            for block in content:
                                if isinstance(block, dict):
                                    if 'toolUse' in block:
                                        tool_use_id = block['toolUse'].get('toolUseId')
                                        if tool_use_id:
                                            tool_use_ids.add(tool_use_id)
                                    elif 'toolResult' in block:
                                        tool_result_id = block['toolResult'].get('toolUseId')
                                        if tool_result_id:
                                            tool_result_ids.add(tool_result_id)
                    
                    orphaned_tool_uses = tool_use_ids - tool_result_ids
                    orphaned_tool_results = tool_result_ids - tool_use_ids
                    
                    if orphaned_tool_uses:
                        logger.warning(f"INVALID STATE: Found {len(orphaned_tool_uses)} orphaned toolUse blocks without matching toolResult:")
                        for tool_id in orphaned_tool_uses:
                            logger.warning(f"  - {tool_id}")
                        logger.warning("This will cause ValidationException with 'Expected toolResult blocks' error")
                        logger.warning("Clearing conversation to reset state...")
                        conversation.messages = []
                        logger.info("Conversation cleared successfully")
                    else:
                        logger.info(f"✓ All {len(tool_use_ids)} toolUse blocks have matching toolResult blocks")
                    
                    if orphaned_tool_results:
                        logger.warning(f"Found {len(orphaned_tool_results)} orphaned toolResult blocks without matching toolUse:")
                        for tool_id in orphaned_tool_results:
                            logger.warning(f"  - {tool_id}")
                    
                    logger.info("=== END VALIDATION ===")
                    
                else:
                    logger.info("Conversation has no 'messages' attribute")
            else:
                logger.info("Agent has no session_manager or conversation")
                
        except Exception as diag_error:
            logger.error(f"Error during conversation diagnostic: {diag_error}")
            import traceback
            logger.error(traceback.format_exc())
        
        logger.info("=== END PRE-INVOCATION DIAGNOSTIC ===")
        
        # Invoke the agent
        result = agent(user_input)

        # Extract response text
        response_text = ""
        if isinstance(result.message, dict) and "content" in result.message:
            if isinstance(result.message["content"], list):
                if len(result.message["content"]) > 0:
                    response_text = result.message["content"][0].get("text", str(result.message))
                else:
                    logger.warning("Agent returned empty content array")
                    response_text = "[Agent returned no content]"
            else:
                response_text = result.message["content"]
        else:
            response_text = str(result.message)
        
        # Add greeting on first invocation for this model if username is available
        greeting = ""
        if first_invocation_per_model.get(model_id, False) and username:
            tool_count = len(tool_names) if tool_names else 5
            greeting = f"Hello, {username}. I'm now using {model_id} with {tool_count} tools.\n"
            first_invocation_per_model[model_id] = False
        
        # Combine greeting and response
        full_response = greeting + response_text
        
        # Extract metrics if available
        usage_metrics = {}
        accumulated_metrics = {}
        if result.metrics:
            logger.info(f"Invocation metrics:")
            
            # Extract usage metrics (tokens)
            if hasattr(result.metrics, 'accumulated_usage') and result.metrics.accumulated_usage:
                usage = result.metrics.accumulated_usage
                usage_metrics = {
                    'inputTokens': usage.get('inputTokens', 0),
                    'outputTokens': usage.get('outputTokens', 0),
                    'totalTokens': usage.get('totalTokens', 0)
                }
                logger.info(f"  Input Tokens: {usage_metrics['inputTokens']}")
                logger.info(f"  Output Tokens: {usage_metrics['outputTokens']}")
                logger.info(f"  Total Tokens: {usage_metrics['totalTokens']}")
            
            # Extract other metrics
            if hasattr(result.metrics, 'accumulated_metrics') and result.metrics.accumulated_metrics:
                accumulated_metrics = result.metrics.accumulated_metrics
                if 'latencyMs' in accumulated_metrics:
                    logger.info(f"  Latency: {accumulated_metrics['latencyMs']} ms")
            
            if hasattr(result.metrics, 'cycle_durations') and result.metrics.cycle_durations:
                logger.info(f"  Cycle Durations: {result.metrics.cycle_durations}")
            
            if hasattr(result.metrics, 'traces') and result.metrics.traces:
                logger.info(f"  Traces: {len(result.metrics.traces)} traces")
        else:
            logger.info("No invocation metrics available")
        
        # Return JSON response with text and metrics
        response_obj = {
            'response': full_response,
            'usage': usage_metrics,
            'metrics': accumulated_metrics
        }
        
        logger.debug(f"Returning JSON response with usage: {usage_metrics}")
        return json.dumps(response_obj)
            
    except Exception as e:
        import traceback
        error_traceback = traceback.format_exc()
        logger.error(f"Error invoking Strands agent: {str(e)}\nTraceback:\n{error_traceback}")
        return f"I encountered an error while processing your request: {str(e)}"

if __name__ == "__main__":
    app.run()