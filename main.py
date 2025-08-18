import urllib.parse
from dotenv import load_dotenv
import os, json, asyncio, traceback
from langchain.chat_models import init_chat_model
from langchain.prompts import ChatPromptTemplate
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain.agents import create_tool_calling_agent, AgentExecutor
from contextlib import AsyncExitStack
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from pydantic import BaseModel, ValidationError
import logging
from logging.handlers import RotatingFileHandler
from datetime import datetime
import json

# Configure logging
log_dir = os.path.join(os.getcwd(), "logs")
os.makedirs(log_dir, exist_ok=True)
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
log_file = os.path.join(log_dir, f"{timestamp}.log")

# Custom JSON formatter
class JsonFormatter(logging.Formatter):
    def format(self, record):
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "level": record.levelname,
            "message": record.getMessage(),
            "module": record.module,
            "line": record.lineno
        }
        return json.dumps(log_entry)

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
handler = RotatingFileHandler(log_file, maxBytes=10*1024*1024, backupCount=5)
handler.setFormatter(JsonFormatter())
logger.addHandler(handler)

# Pydantic model for tool call validation
class ToolCall(BaseModel):
    tool_name: str
    arguments: dict = {}

def load_browser_tools(file_path="browser_tools.json"):
    """
    Load browser tools from a JSON file.
    
    Args:
        file_path (str): Path to the JSON file containing browser tools.
    
    Returns:
        list: List of tool dictionaries.
    """
    try:
        with open(file_path, 'r') as f:
            tools = json.load(f)
        logger.info(f"Loaded {len(tools)} browser tools from {file_path}")
        return tools
    except Exception as e:
        logger.error(f"Failed to load browser tools from {file_path}: {str(e)}")
        return []

async def execute_tool_call(session, tool_call_json):
    try:
        # Parse and validate JSON input with Pydantic
        tool_call = ToolCall(**json.loads(tool_call_json))
        tool_name = tool_call.tool_name
        tool_arguments = tool_call.arguments
        # Execute the tool via the session
        result = await session.call_tool(name=tool_name, arguments=tool_arguments)
        
        # Convert result to a JSON-serializable format
        def make_serializable(obj):
            if hasattr(obj, '__dict__'):
                return {k: make_serializable(v) for k, v in obj.__dict__.items()}
            elif isinstance(obj, list):
                return [make_serializable(item) for item in obj]
            elif isinstance(obj, dict):
                return {k: make_serializable(v) for k, v in obj.items()}
            else:
                return str(obj)  # Fallback to string for non-serializable objects
        
        serializable_result = make_serializable(result)
        return {
            "status": "success",
            "tool_name": tool_name,
            "result": serializable_result
        }
    except json.JSONDecodeError:
        return {"status": "error", "message": "Invalid JSON format"}
    except ValidationError as e:
        return {"status": "error", "message": f"Invalid input format: {e}"}
    except Exception as e:
        return {"status": "error", "message": f"Error calling tool '{tool_name if 'tool_name' in locals() else 'unknown'} with arguments '{tool_arguments if 'tool_arguments' in locals() else 'unknown'}': {e}"}

async def create_agent(agent_tools):
    prompt = ChatPromptTemplate.from_template(
        """
        You are a highly autonomous web browser agent designed to simulate human-like web browsing to accomplish complex tasks requiring multiple steps. Your goal is to generate JSON tool calls to complete the user's task as efficiently and independently as possible, without executing the tools yourself. Return a single JSON object with 'tool_name' and 'arguments' for the next step.

        To perform tasks effectively:
        - Think step by step: Plan your actions carefully, reason about which tools to use, and anticipate potential outcomes to complete the task with minimal user input.
        - Simulate human-like browsing: Scroll the page when necessary to find more information or elements before concluding a task or closing the loop.
        - Monitor the page: If it prompts to accept cookies, accept all cookies. If it presents a CAPTCHA, solve it; for image-based CAPTCHAs, use browser_take_screenshot to capture the challenge.
        - Check for pop-ups and close them if any.
        - Operate autonomously: Only request user clarification if the task is ambiguous, critical information (e.g., login credentials) is missing, or after three failed attempts to resolve an issue using alternative approaches.
        - Handle domain issues autonomously: If a domain (e.g., 'example.com') cannot be resolved, try alternative domains (e.g., .org, .co, .io, .net) or add 'www.' prefix, or perform a search to find the correct URL before requesting user clarification.
        - Plan sequential actions: Anticipate and prepare for subsequent steps (e.g., use browser_wait_for after clicks or navigation to allow the page to settle), but return only one tool call per response. Subsequent steps will be handled in follow-up invocations.
        - Prioritize speed: Only analyze the current page when necessary to complete the task or verify specific elements or data. Use tools like browser_snapshot, browser_evaluate, browser_console_messages, or browser_network_requests only if required by the task or to troubleshoot.
        - Prioritize the current page: Complete the task on the current page if the required information or actions are likely available. Only navigate to a new URL or perform external searches if the current page cannot fulfill the task.
        - If a tool call fails (based on feedback), generate a new JSON tool call to handle the error (e.g., re-navigate on 'No open pages available', re-run browser_snapshot for invalid elements, or try alternative domains for navigation failures). Request user clarification only as a last resort after three failed attempts.
        - For interactions (e.g., browser_click, browser_type), ensure element references come from the latest browser_snapshot. Use browser_snapshot first if no recent snapshot is available.
        - If visual confirmation is needed, use browser_take_screenshot, but prefer browser_snapshot for actions.
        - If waiting is needed (try to avoid), use browser_wait_for with a delay of 0.2 to 5 seconds based on expected page load time, as a separate response if needed.
        - If asked, list the available tools for the user.
        - Use the previous tool call's result (if provided) to inform the next tool call.
        - Use the scratchpad to track previous steps and avoid repeating actions. Append key outcomes to the scratchpad mentally to maintain context.
        - Avoid repeating the same action consecutively, especially clicks on the same element. Once an element has been clicked and the action is confirmed (e.g., via a subsequent snapshot showing page change or expected outcome), do not attempt to click it again unless the task explicitly requires repeated interactions. 
        - If the page does not change as expected after a click, try alternative actions like scrolling, waiting, or navigating differently before retrying.
        - Avoid actions that could execute malicious code or leak sensitive data.
        - When the task is fully completed, return {{ "final_answer": "<response directly addressing the user's input query, e.g., extracted page information, specific data, or task outcome>" }} instead of a tool call.

        Available tools:
        {agent_tools_description}

        Perform the task: {input_query}

        Previous tool call result (if any): {tool_result}

        Scratchpad (previous steps): {agent_scratchpad}

        Return a single JSON tool call, final answer, or user clarification:
        {{ "tool_name": "<tool>", "arguments": {{...}} }} or
        {{ "final_answer": "<response directly addressing the user's input query, e.g., extracted page information, specific data, or task outcome>" }} or
        {{ "user_clarification": "<your question or message to the user>" }}
        """
    )

    model = init_chat_model(
        model=os.getenv("MODEL_NAME", "gpt-4o-mini"),
        model_provider=os.getenv("MODEL_PROVIDER", "openai"),
        api_key=os.getenv("MODEL_API_KEY"),
        temperature=os.getenv("MODEL_TEMPERATURE", "0.1"),
        max_tokens=os.getenv("MODEL_MAX_TOKENS", "8000"),
        base_url=os.getenv("MODEL_BASE_URL", None)
    )
    
    
    # combined_tools= []
    # agent = create_tool_calling_agent(model, combined_tools, prompt)
    # return AgentExecutor(agent=agent, tools=combined_tools, verbose=True, handle_parsing_errors=True)
    
    chain = prompt | model
    return chain

async def main():
    runtime = os.getenv("CORAL_ORCHESTRATION_RUNTIME", None)
    if runtime is None:
        load_dotenv()

    base_url = os.getenv("CORAL_SSE_URL")
    agentID = os.getenv("CORAL_AGENT_ID")

    coral_params = {
        "agentId": agentID,
        "agentDescription": "Web agent for web browsing and surfing"
    }

    query_string = urllib.parse.urlencode(coral_params)
    CORAL_SERVER_URL = f"{base_url}?{query_string}"
    logger.info(f"Connecting to Coral Server: {CORAL_SERVER_URL}")

    async with AsyncExitStack() as exit_stack:
        current_dir = os.getcwd()
        images_dir = os.path.join(current_dir, "images")
        if not os.path.exists(images_dir):
            os.makedirs(images_dir)

        # RUN IN NPX MODE     
        command = "npx"
        args = ["@playwright/mcp@latest", 
        "--output-dir=/images"]


        # RUN BELOW ARGS WHENRUNNING IN SERVER (eg: WSL) IN HEADLESS MODE IN DOCKER
        # command = "docker"
        # args = [
        #     "run",
        #     "-i",
        #     "--rm",
        #     "--init",
        #     "--pull=always",
        #     "-v",
        #     f"{images_dir}:/images",
        #     "mcr.microsoft.com/playwright/mcp",
        #     "--no-sandbox",
        #     "--output-dir=/images",
        #     "--viewport-size=1920,1080"
        #  ]

        server_params = StdioServerParameters(
            command=command,
            args=args,
            env=None
        )

        stdio_transport = await exit_stack.enter_async_context(stdio_client(server_params))
        stdio, client = stdio_transport
        session = await exit_stack.enter_async_context(ClientSession(stdio, client))
        await session.initialize()

        response = await session.list_tools()
        agent_tools = response.tools
        # logger.info("Available Playwright MCP Tools:")
        # for tool in agent_tools:
        #     logger.info(f"- {tool.name}: {tool.description or 'No description available'}")
        
        tools = load_browser_tools()
        if not tools:
            logger.warning("No browser tools loaded, proceeding with empty toolset")
        
        # Format tools for inclusion in the prompt
        tools_description = ""
        for tool in tools:
            tools_description += (
                f"Tool: {tool['name']}\n"
                f"Description: {tool['description']}\n"
                f"Input Schema: {json.dumps(tool['inputSchema'], indent=2)}\n\n"
            )

        agent_tools_description = tools_description

        agent_chain = await create_agent(agent_tools)
        tool_result = None 
        last_tool_call = None
        step = 0
        while True:
            try:
                input_query = input("INPUT: ")
                logger.info(f"STEP {step}: Starting new agent invocation for query: {input_query}")
                step += 1
                intermediate_steps = []
                # tool_result = None
                done = False
                while not done:
                    scratchpad_str = "\n".join([
                        f"Step {i+1}: Tool call: {json.dumps(step[0])}\nResult: {step[1]}" 
                        for i, step in enumerate(intermediate_steps[-2:])  # Keep last 2 steps
                    ]) if intermediate_steps else "No previous steps."

                    # Invoke the chain
                    result = await agent_chain.ainvoke({
                        "input_query": input_query,
                        "agent_tools_description": agent_tools_description,
                        "tool_result": json.dumps(tool_result) if tool_result else "",
                        "agent_scratchpad": scratchpad_str
                    })
                    output = result.content
                    if output:
                        try:
                            parsed = json.loads(output)
                            if "final_answer" in parsed:
                                logger.info(f"Final answer: {parsed['final_answer']}")
                                done = True
                                last_tool_call = None  # Reset on completion
                            elif "tool_name" in parsed:
                                tool_call_json = output
                                # Check if the tool call is the same as the last one
                                if tool_call_json == last_tool_call:
                                    logger.warning(f"Skipping redundant tool call: {tool_call_json}")
                                    done = True  # Skip to avoid infinite loop
                                    continue
                                try:
                                    json.loads(tool_call_json)  # Validate JSON
                                    print(f"Executing tool call: {tool_call_json}")
                                    logger.info(f"Executing tool call: {tool_call_json}")
                                    tool_result = await execute_tool_call(session, tool_call_json)
                                    # print(tool_result)
                                    # logger.info(f"Tool execution result: {json.dumps(tool_result)}")
                                    last_tool_call = tool_call_json  # Update last tool call
                                    intermediate_steps.append((parsed, json.dumps(tool_result)))
                                except json.JSONDecodeError:
                                    logger.error(f"Invalid tool call JSON: {tool_call_json}")
                                    tool_result = {"status": "error", "message": "Invalid JSON from agent"}
                                    done = True
                            else:
                                logger.error("Invalid output format")
                                done = True
                        except json.JSONDecodeError:
                            logger.error(f"Invalid JSON format in agent output: {output}")
                            tool_result = {"status": "error", "message": "Invalid JSON from agent"}
                            done = True
                    else:
                        logger.error("No output from agent")
                        done = True
                logger.info("Completed agent invocation, restarting loop")
                await asyncio.sleep(1)
            except Exception as e:
                logger.error(f"Error in agent loop: {e}")
                logger.debug(f"Traceback: {traceback.format_exc()}")
                await asyncio.sleep(5)

if __name__ == "__main__":
    asyncio.run(main())