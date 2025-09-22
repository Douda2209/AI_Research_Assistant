# Here we write the tools that our Agent/Model can use to return the response
# > pip install -U ddgs
## Tools for Wikipedia and DuckDuck Search
from langchain_community.tools import WikipediaQueryRun, DuckDuckGoSearchRun
from langchain_community.utilities import WikipediaAPIWrapper
from langchain.tools import Tool
from datetime import datetime

search = DuckDuckGoSearchRun()
search_tool = Tool(
    name="searchWebTool", #Name of Tool
    func=search.run,
    description="Search the web for information"

)
# pass the tool or added to the tools_list and give it to the agent so he uses it

# Wikipedia tool:
api_warpper = WikipediaAPIWrapper(top_k_results=2, doc_content_chars_max=100)
wiki_tool = WikipediaQueryRun(api_wrapper=api_warpper)

## own cunstom Tool:
# save the response to a .txt file:
def save_to_txt(data:str, filename:str ="Research_Output.txt"):
    timestamp = datetime.now().strftime("%D-%M-%Y-%d %H:%M:%S")
    formatted_text = f"---Research Output ---\nTimestamp: {timestamp}\n\n{data}\n\n"

    with open(filename, "a", encoding="utf-8") as f:
        f.write(formatted_text)

    return f"Data successfully saved to {filename}"

# wrapp it: # when passing the query to the agent, tasks him to save it into txt file inside the prompt and then he knows that there is a tool he can use it
save_tool = Tool(
    name="save_text_to_file",
    func = save_to_txt,
    description="saves structured resarch data to a text file",
)