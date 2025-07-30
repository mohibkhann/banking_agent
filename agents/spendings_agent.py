import pandas as pd
from langchain_openai import ChatOpenAI
from langchain.tools import tool

from langgraph.graph import StateGraph
from dotenv import load_dotenv

load_dotenv()

df = pd.read_csv("C:/Users/mohib.alikhan/Desktop/Banking-Agent/Banking_Data.csv")



if __name__ == "__main__":
    print("Hello this is spendings agent")