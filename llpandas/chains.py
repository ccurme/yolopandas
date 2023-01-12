from langchain.chains import LLMChain
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate


llm = OpenAI(temperature=0)
template = """
You are working with a pandas dataframe in Python. The name of the dataframe is `df`.

You should execute code as commanded to either provide information to answer the question or to
do the transformations required.

You should not assign any variables; you should return a one-liner in Pandas.

This is your objective: {objective}

Go!

```python
print(df.head())
```
```output
llm_result = {df_head}
```
```python"""
PROMPT = PromptTemplate(template=template, input_variables=["objective", "df_head"])
LLM_CHAIN = LLMChain(llm=llm, prompt=PROMPT)
