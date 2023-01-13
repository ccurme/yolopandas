import ast
from typing import Any

from IPython.display import clear_output
from langchain.chains.base import Chain
from langchain.input import print_text
import pandas as pd

from llpandas.chains import LLM_CHAIN, LLM_CHAIN_WITH_MEMORY


@pd.api.extensions.register_dataframe_accessor("llm")
class LLMAccessor:
    def __init__(self, pandas_df: pd.DataFrame):
        self.df = pandas_df

    def _query(self, query: str, chain: Chain = LLM_CHAIN, verify: bool = True) -> Any:
        """Query the dataframe using a specified Chain."""
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
            exec(ast.unparse(module))
            module_end = ast.Module(tree.body[-1:], type_ignores=[])
            module_end_str = ast.unparse(module_end)
            try:
                return eval(module_end_str)
            except Exception:
                exec(module_end_str)

    def query(self, query: str, verify: bool = True) -> Any:
        """Query the dataframe with natural language."""
        return self._query(query, chain=LLM_CHAIN, verify=verify)

    def query_with_memory(self, query: str, verify: bool = True) -> Any:
        """Query the dataframe with natural language. Retain history of conversation."""
        return self._query(query, chain=LLM_CHAIN_WITH_MEMORY, verify=verify)
