from langchain import OpenAI, LLMChain, PromptTemplate
from langchain.chains.conversation.memory import ConversationBufferMemory


LLM = OpenAI(temperature=0)
TEMPLATE = """
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
PROMPT = PromptTemplate(
    template=TEMPLATE, input_variables=["chat_history", "query", "df_head"]
)
MEMORY = ConversationBufferMemory(memory_key="chat_history", input_key="query")
LLM_CHAIN = LLMChain(llm=LLM, prompt=PROMPT, memory=MEMORY)
