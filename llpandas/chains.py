import os

from langchain import LLMChain, OpenAI, PromptTemplate
from langchain.chains.conversation.memory import ConversationBufferMemory
from langchain.llms.loading import load_llm

LLM_CONFIGURATION_PATH = os.environ.get("LLPANDAS_LLM_CONFIGURATION")
if LLM_CONFIGURATION_PATH is None:
    LLM = OpenAI(temperature=0)
else:
    LLM = load_llm(LLM_CONFIGURATION_PATH)


# Default template, no memory
TEMPLATE = """
You are working with a pandas dataframe in Python. The name of the dataframe is `df`.

You should execute code as commanded to either provide information to answer the question or to
do the transformations required.

You should not assign any variables; you should return a one-liner in Pandas.

This is your objective: {query}

Go!

```python
print(df.head())
```
```output
{df_head}
```
```python"""

PROMPT = PromptTemplate(template=TEMPLATE, input_variables=["query", "df_head"])
LLM_CHAIN = LLMChain(llm=LLM, prompt=PROMPT)


# Template with memory
# TODO: add result of exected code to memory; currently we only remember what code was run.
TEMPLATE_WITH_MEMORY = """
You are working with a pandas dataframe in Python. The name of the dataframe is `df`.

You are interacting with a programmer. The programmer issues commands and you should translate
them into Python code and execute them.

This is the history of your interaction so far:
{chat_history}

Human: {query}

Go!

```python
df.head()
```
```output
{df_head}
```
```python
"""
PROMPT_WITH_MEMORY = PromptTemplate(
    template=TEMPLATE_WITH_MEMORY, input_variables=["chat_history", "query", "df_head"]
)
MEMORY = ConversationBufferMemory(memory_key="chat_history", input_key="query")
LLM_CHAIN_WITH_MEMORY = LLMChain(llm=LLM, prompt=PROMPT_WITH_MEMORY, memory=MEMORY)
