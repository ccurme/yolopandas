# YOLOPandas

Interact with Pandas objects via LLMs and [LangChain](https://github.com/hwchase17/langchain).

YOLOPandas lets you specify commands with natural language and execute them directly on Pandas objects.
You can preview the code before executing, or set `yolo=True` to execute the code straight from the LLM.

**Warning**: YOLOPandas will execute arbitrary Python code on the machine it runs on. This is a dangerous thing to do.

https://user-images.githubusercontent.com/26529506/214591990-c295a283-b9e6-4775-81e4-28917183ebb1.mp4

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

`.query` can return the result of the computation, which we do not constrain. For instance, while `"Show me products under $10"` will return a dataframe, the query `"Split the dataframe into two, 1/3 in one, 2/3 in the other. Return (df1, df2)"` can return a tuple of two dataframes. You can also chain queries together, for instance:
```python
df.llm.query("Group by type and take the mean of all numeric columns.", yolo=True).llm.query("Make a bar plot of the result and use a log scale.", yolo=True)
```

See the [example notebook](docs/example_notebooks/example.ipynb) for more ideas.


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

By working with LangChain's Chain abstraction, it is very easy to plug-and-play different chains into YOLOPandas. This can be useful if you want to customize the prompt, customize the chain, or anything like that.

To use a custom chain for a particular dataframe, you can do:

```python
df.set_chain(chain)
```

If you ever want to reset the chain to the base chain, you can do:

```python
df.reset_chain()
```

### Memory Abstraction

The default chain used by YOLOPandas utilizes the LangChain concept of [memory](https://langchain.readthedocs.io/en/latest/modules/memory.html). This allows for "remembering" of previous commands, making it possible to ask follow up questions or ask for execution of commands that stem from previous interactions.

For example, the query `"Make a seaborn plot of price grouped by type"` can be followed with `"Can you use a dark theme, and pastel colors?"` upon viewing the initial result.

By default, memory is turned on. In order to have it turned off by default, you can set the environment variable `LLPANDAS_USE_MEMORY=False`.

If you are resetting the chain, you can also specify whether to use memory there:

```python
df.reset_chain(use_memory=False)
```


