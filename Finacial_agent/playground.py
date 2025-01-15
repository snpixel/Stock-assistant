from phi.agent import Agent
import phi.api
from phi.tools.yfinance import YFinanceTools
from phi.tools.duckduckgo import DuckDuckGo
from phi.model.groq import Groq
from dotenv import load_dotenv
import os 
import phi
from phi.playground import Playground,serve_playground_app

load_dotenv()

phi.api = os.getenv("PHI_API_KEY")

web_search_agent = Agent(
    name="Web Search Agent",
    role="Search the web for the information",
    model = Groq(id="llama-3.1-70b-versatile"),
    tools=[DuckDuckGo()],
    instructions=["Always include sources"],
    show_tool_calls=True,
    markdown=True,
)

# financial agent
finance_agent = Agent(
    name="Financial AI Agent",
    model=Groq(id="llama-3.1-70b-versatile"),
    tools=[
        YFinanceTools(stock_price=True, stock_fundamentals=True,analyst_recommendations=True,
                    company_news=True)
        ],
    instructions=["Use Tables to display the data"],
    show_tool_calls=True,
    markdown=True
    )

app = Playground(agents=[web_search_agent,finance_agent]).get_app()

if __name__=="__main__":
    serve_playground_app("playground:app",reload=True)