#pip install pygpt4all
#https://github.com/ggerganov/llama.cpp/blob/460c48254098b28d422382a2bbff6a0b3d7f7e17/main.cpp#L794

import sqlite3
import pandas as pd
from sklearn import datasets

# Load iris dataset
iris = datasets.load_iris()
df = pd.DataFrame(data=iris.data, columns=iris.feature_names).head(20)
print(df.shape)
print(df.head())

# Create SQLite database and cursor
conn = sqlite3.connect('iris.db')
c = conn.cursor()

# Insert iris data into table
df.to_sql('iris', conn, if_exists='replace', index=False)

# Close connection
conn.close()

from langchain.agents import create_sql_agent
from langchain.agents.agent_toolkits import SQLDatabaseToolkit
from langchain.sql_database import SQLDatabase
from langchain.agents import AgentExecutor
from langchain.llms import OpenAI
from langchain.llms import LlamaCpp
import urllib
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.llms import GPT4All

db = SQLDatabase.from_uri('sqlite:///iris.db')

## Cloud
# model = OpenAI()

## Edge
model = LlamaCpp(model_path="./model/GPT4All-13B-snoozy.ggml.q5_0.bin", verbose=True, n_ctx=1024, temperature=0)
# model = LlamaCpp(model_path="./models/ggml-vicuna-7b-4bit-rev1.bin", verbose=True, n_threads=16)
# model = LlamaCpp(model_path="./models/ggml-vicuna-13b-4bit-rev1.bin", verbose=True, n_threads=16)

toolkit = SQLDatabaseToolkit(db=db,llm=model)

# Invoke the agent
agent_executor = create_sql_agent(
    llm=model,
    toolkit=toolkit,
    verbose=True
)

agent_executor.run("Describe the iris table")
agent_executor.run("In the iris table, what is the average sepal length?")
# agent_executor.run("How many people were injured in total?")
# agent_executor.run("How many people suffered a wound?") #OpenAI
# agent_executor.run("Give me a row from RadioMessages")
