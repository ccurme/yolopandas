import ast
import os
from typing import Any, Optional

import pandas as pd
from IPython.display import clear_output
from langchain.chains.base import Chain
from langchain.input import print_text
from langchain.llms.base import BaseLLM

from yolopandas.chains import get_chain


@pd.api.extensions.register_dataframe_accessor("llm")
class LLMAccessor:
    def __init__(self, pandas_df: pd.DataFrame):
        self.df = pandas_df
        use_memory = bool(os.environ.get("LLPANDAS_USE_MEMORY", True))
        self.chain = get_chain(use_memory=use_memory)

    def set_chain(self, chain: Chain) -> None:
        """Set chain to use."""
        self.chain = chain

    def reset_chain(
        self, llm: Optional[BaseLLM] = None, use_memory: bool = True
    ) -> None:
        """Reset chain with LLM or memory kwarg."""
        self.chain = get_chain(llm=llm, use_memory=use_memory)

    def query(self, query: str, yolo: bool = False) -> Any:
        """Query the dataframe."""
        df = self.df
        df_columns = df.columns.tolist()
        inputs = {"query": query, "df_head": df.head(), "df_columns": df_columns, "stop": "```"}
        llm_response = self.chain.run(**inputs)
        eval_expression = False
        if not yolo:
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
