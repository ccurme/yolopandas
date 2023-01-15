import ast
from typing import Any

import pandas as pd
from IPython.display import clear_output
from langchain.input import print_text

from llpandas.chains import LLM_CHAIN


@pd.api.extensions.register_dataframe_accessor("llm")
class LLMAccessor:
    def __init__(self, pandas_df: pd.DataFrame):
        self.df = pandas_df

    def query(self, query: str, verify: bool = True) -> Any:
        """Query the dataframe with natural language."""
        df = self.df
        inputs = {"objective": query, "df_head": df.head(), "stop": ["```"]}
        llm_response = LLM_CHAIN.run(**inputs)
        eval_expression = False
        if verify:
            print("suggested code:")
            print(llm_response)
            print("run this code? y/n")
            user_input = input()
            if user_input == "y":
                clear_output(wait=True)
                print_text(llm_response, color="green")
                eval_expression = True
        else:
            eval_expression = True

        if eval_expression:
            # WARNING: This is a bad idea. Here we evaluate the (potentially multi-line)
            # llm response. Do not use unless you trust that llm_response is not malicious.
            tree = ast.parse(llm_response)
            module = ast.Module(tree.body[:-1], type_ignores=[])
            expression = ast.Expression(tree.body[-1].value)
            exec(ast.unparse(module))
            module_end = ast.Module(tree.body[-1:], type_ignores=[])
            module_end_str = ast.unparse(module_end)
            try:
                return eval(module_end_str)
            except:
                exec(module_end_str)
