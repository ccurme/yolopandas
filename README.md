# YOLOPandas

Interact with Pandas objects via LLMs and [langchain](https://github.com/hwchase17/langchain).

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
