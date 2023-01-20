# YOLOPandas

Interact with Pandas objects via LLMs and [LangChain](https://github.com/hwchase17/langchain).

YOLOPandas lets you specify commands with natural language and execute them directly on Pandas objects.
You can preview the code before executing, or set `yolo=True` to execute the code straight from the LLM.

**Warning**: YOLOPandas will execute arbitrary Python code on the machine it runs on. This is a dangerous thing to do.

## Quick Install

`pip install yolopandas`

## Basic usage

YOLOPandas adds a `llm` accessor to Pandas dataframes.

```python
from yolopandas import pd

df = pd.DataFrame(
    [
        {"name": "The Da Vinci Code", "type": "book", "price": 15, "quantity": 300, "rating": 4},
        {"name": "Jurassic Park", "type": "book", "price": 12, "quantity": 400, "rating": 4.5},
        {"name": "Jurassic Park", "type": "film", "price": 8, "quantity": 6, "rating": 5},
        {"name": "Matilda", "type": "book", "price": 5, "quantity": 80, "rating": 4},
        {"name": "Clockwork Orange", "type": None, "price": None, "quantity": 20, "rating": 4},
        {"name": "Walden", "type": None, "price": None, "quantity": 100, "rating": 4.5},
    ],
)

df.llm.query("What item is the least expensive?")
```
The above will generate Pandas code to answer the question, and prompt the user to accept or reject the proposed code.
Accepting it in this case will return a Pandas dataframe containing the result.

Alternatively, you can execute the LLM output without first previewing it:
```python
df.llm.query("What item is the least expensive?", yolo=True)
```

## LangChain Components

This package uses several LangChain components, making it easy to work with if you are familiar with LangChain. In particular, it utilizes the LLM, Chain, and Memory abstractions.

### LLM Abstraction

By working with LangChain's LLM abstraction, it is very easy to plug-and-play different LLM providers into YOLOPandas. You can do this in a few different ways:

1. You can change the default LLM by specifying a config path using the `LLPANDAS_LLM_CONFIGURATION` environment variable. The file at this path should be in [one of the accepted formats](https://langchain.readthedocs.io/en/latest/modules/llms/examples/llm_serialization.html).

2. If you have a LangChain LLM wrapper in memory, you can set it as the default LLM to use by doing:

```python
import yolopandas
yolopandas.set_llm(llm)
```

3. You can set the LLM wrapper to use for a specific dataframe by doing: `df.reset_chain(llm=llm)`


### Chain Abstraction

### Memory Abstraction


