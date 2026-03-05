# Bedrock AgentCore Strands Playground

Read [this document](./agentcore_playground.md) for background.

## Features

- **Dynamic tool and model selection:** change available tools and model on the fly
- **Real-time Chat Interface**: Interactive chat with deployed AgentCore agents
- **Agent Discovery**: Automatically discover and select from available agents in your AWS account
- **Version Management**: Choose specific versions of your deployed agents
- **Multi-Region Support**: Connect to agents deployed in different AWS regions
- **Streaming Responses**: Real-time streaming of agent responses
- **Session Management**: Maintain conversation context with unique session IDs
- **Memory Management**: Uses AgentCore Memory to save user preferences and conversation context

## Architecture

![AgentCore Architecture](./images/BRAC_architecture.png)

## Prerequisites

- Python 3.11 or higher
- [uv package manager](https://docs.astral.sh/uv/getting-started/installation/)
- AWS CLI configured with appropriate credentials
- Access to Amazon Bedrock AgentCore service
- Deployed agents on Bedrock AgentCore Runtime
- Optional: Cognito user pool (only if you need authentication - disabled by default)

### Required AWS Permissions

Your AWS credentials need the following permissions:

- `bedrock-agentcore-control:ListAgentRuntimes`
- `bedrock-agentcore-control:ListAgentRuntimeVersions`
- `bedrock-agentcore:InvokeAgentRuntime`
  Note that other permissions may be required by tools you select.

## Installation

1. **Clone the repository**:

   ```bash
   git clone https://github.com/aws-samples/aws-generativeai-partner-samples/tree/main/agentcore-strands-playground
   ```
2. **Install dependencies using uv**:

   ```bash
   uv sync
   ```
3. **Activate the venv**:
   ```bash
   source .venv/bin/activate
   ```

## Deploy the Example Agent

1. **Configure the agent**:
```bash
cd agentcore_agent
uv run agentcore configure -e runtime_agent.py
```
Select default options and configure Long Term Memory under Memory Configuration. Authentication is disabled by default.

2. **Deploy to AgentCore Runtime:**
```bash
uv run agentcore launch
cd ..
```

3. **Optional: Set up Cognito pool (only if you need authentication)**
   
   If you need authentication, run 'config.py --cognito' to configure a Cognito pool automatically. Then use '--auth' when running the app.
## Running the Application

### Using uv (recommended)
```bash
uv run streamlit run app.py
```
The application will start and be available at `http://localhost:8501`.

Note: Streamlit apps run with an unencrypted, unauthenticated endpoint. Do not run the app in a public subnet in your VPC.

It can be useful to add policies to the IAM role used by the agent. For example, to grant permission to the "use_aws" tool to read S3, attach the AmazonS3ReadOnlyAccess policy:

```bash
aws iam attach-role-policy \
    --role-name AmazonBedrockAgentCoreSDKRuntime-us-west-2-xxxxxx \
    --policy-arn arn:aws:iam::aws:policy/AmazonS3ReadOnlyAccess
```

## Usage

Note: all parameters have defaults, which typically pick the most recent agent/version/session. The simplest usage is to run the front-end application and start chatting with the agent.

Optional:

1. **Configure Tools:** Click the Tools Configuration dropdown to select from available Strands Agents built-in tools
2. **Configure AWS Region**: Select your preferred AWS region from the sidebar
3. **Select Agent**: Choose from automatically discovered agents in your account
4. **Choose Version**: Select the specific version of your agent to use
5. **Select Memory**: The front-end will discover configured AgentCore Memory resources and select the most recently created
6. **Select Session:** Choose from a saved session or enter a new session name and click "New session"
7. **Select Tools:**
8. **Start Chatting**: Type your message in the chat input and press Enter

## Project Structure

```
agentcore-strands-playground/
├── app.py                           # Main Streamlit application
├── br_utils.py                      # Bedrock utilities (model discovery)
├── config.py                        # AWS resource configuration script
├── cleanup.py                       # AWS resource cleanup script
├── dotenv.example                   # Example environment variables
├── pyproject.toml                   # Project dependencies (uv)
├── README.md                        # This file
├── agentcore_agent/                 # Agent deployment configuration
│   ├── runtime_agent.py             # Strands agent implementation
│   ├── requirements.txt             # Agent dependencies
├── images/                          # Documentation images
│   ├── BRAC_architecture.png
│   ├── BRAC_interface_screen.png
│   └── ...
└── static/                          # UI assets (fonts, icons, logos)
    ├── agentcore-service-icon.png
    ├── gen-ai-dark.svg
    ├── user-profile.svg
    └── ...
```

## Configuration Files

- **`pyproject.toml`**: Defines project dependencies and metadata

## Troubleshooting

### Common Issues

1. **No agents found**: Ensure you have deployed agents in the selected region and have proper AWS permissions
2. **Connection errors**: Verify your AWS credentials and network connectivity
3. **Permission denied**: Check that your IAM user/role has the required Bedrock AgentCore permissions

### Debug Mode

Enable debug logging by setting the Streamlit logger level in the application or check the browser console for additional error information.

## Development

### Adding New Features

The application is built with modularity in mind, particularly the ability for AWS partners to add funcationality. Key areas for extension:

- **Partner LLMs**: the list of LLMs returned in br_utils.py can be modified
- **Observability**: the agent is configured to report OTEL logs which can be consumed by partner observability solutions
- **Identity**: the auth_utils.py module can be replaced by a partner IdP solution
- **Partner Memory**: the memory interface can be modified to use different partner tools
- **MCP Servers**: any partner MCP server can be called by the agent

### Dependencies

- **boto3**: AWS SDK for Python
- **streamlit**: Web application framework
- **uv**: Fast Python package installer and resolver

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## License

This project is licensed under the terms specified in the repository license file.

## Resources

- [Amazon Bedrock AgentCore Documentation](https://docs.aws.amazon.com/bedrock-agentcore/)
- [Bedrock AgentCore Samples](https://github.com/awslabs/amazon-bedrock-agentcore-samples/)
- [Streamlit Documentation](https://docs.streamlit.io/)
- [Strands Agents Framework](https://github.com/awslabs/strands-agents)
  Agent
