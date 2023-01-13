import ast
from typing import Any

from langchain.chains.base import Chain
import pandas as pd

from llpandas.chains import LLM_CHAIN


@pd.api.extensions.register_dataframe_accessor("llm")
class LLMAccessor:
    def __init__(self, pandas_df: pd.DataFrame):
        self.df = pandas_df

    def query(self, query: str, chain: Chain = LLM_CHAIN, verify: bool = True) -> Any:
        """Query the dataframe with natural language."""
        df = self.df
        inputs = {"query": query, "df_head": df.head(), "stop": "```"}
        llm_response = chain.run(**inputs)
        eval_expression = False
        if verify:
            print("suggested code:")
            print(llm_response)
            print("run this code? y/n")
            user_input = input()
            if user_input == "y":
                eval_expression = True
        else:
            eval_expression = True

        if eval_expression:
            # WARNING: This is a bad idea. Here we evaluate the (potentially multi-line)
            # llm response. Do not use unless you trust that llm_response is not malicious.
            # Adapted from https://stackoverflow.com/a/41472638
            tree = ast.parse(llm_response)
            module = ast.Module(tree.body[:-1], type_ignores=[])
            expression = ast.Expression(tree.body[-1].value)
            exec(compile(module, "", "exec"))
            return eval(compile(expression, "", "eval"))