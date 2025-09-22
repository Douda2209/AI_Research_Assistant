from dotenv import load_dotenv
from pydantic import BaseModel
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from tools import search_tool, wiki_tool, save_tool
load_dotenv()

# instatiate a chat model:
# choice 1: OpenAI "gpt-5" for chat or "o4-mini-deep-research" for the specific deep research
#llm = ChatOpenAI(model="gpt-4o-mini")

#choice 2: Anthropic
#llm2 = ChatAnthropic(model= "claude-3-7-sonnet-20250219")

#choice 3: gemini

from langchain.chat_models import init_chat_model
llm3 = init_chat_model("gemini-2.5-flash", model_provider="google_genai")

# testing the llm and API_KEY:
#response = llm3.invoke("What is a graph theory?")
#print(response) #OK

# Create prompt template:

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser

## we define a class to specify the type of content that the llm should generate
## llm takes a prompt and llm generates a response using the schema of the class we defined

# provide the tools imported from tools.py and pass it to the agent and in agent executor:

tools_from_module = [search_tool, wiki_tool, save_tool]
class ResearchResponse(BaseModel): #inherit from BaseModel class and we can add whatever we want
    # define the attributes:
    topic:str
    summary:str
    sources:list[str]
    tools_used:list[str]

# instatiate a parser: can also be done with json
## parser takes the output of the llm and parses it into the model class
parser = PydanticOutputParser(pydantic_object=ResearchResponse)

# define the prompt:
prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system", #1-define the ai-agent by giving it its role, 2- main task, 3- how to reponse
            """
            You are a research assistant that will help generate a research paper. 
            Answer the user query and use the necessary tools.
            Wrap the output in this format and provide no other text\n{format_instructions}
            """,
        ),
        ("placeholder", "{chat_history}"),
        ("human", "{query}"),
        ("placeholder","{agent_scratchpad}"),
    ]
).partial(format_instructions=parser.get_format_instructions()) # parser takes the model, parses it as sting and gives it to the llm as part of the system prompt

# define the agent:
from langchain.agents import create_tool_calling_agent, AgentExecutor

agent = create_tool_calling_agent(
    llm3,
    prompt=prompt,
    tools=tools_from_module

)
# pass the tools_list to the executor:
agent_executor = AgentExecutor(agent=agent, tools= tools_from_module, verbose=True)

# unstructured reponse raw:
# we can get an user_input query/question:
query_input = input("What is your research question?")

# raw_response = agent_executor.invoke({"query": "what is the biggest airplane?"}) #query is from the model
# user_input query:
raw_response = agent_executor.invoke({"query": query_input}) #query is from the model
#print(raw_response)

# structured response: using the parser:
structured_response = parser.parse(raw_response.get("output")) # because the raw respone is a dict {"query": str, "output": array[str]

try:
    print(structured_response.summary)
except Exception as e:
    print("Error parsing the response: ", e, "RAW RESPONSE: \n", raw_response )

