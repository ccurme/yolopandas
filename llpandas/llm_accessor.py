from typing import Any

import pandas as pd

from llpandas.chains import LLM_CHAIN


@pd.api.extensions.register_dataframe_accessor("llm")
class LLMAccessor:

    def __init__(self, pandas_df: pd.DataFrame):
        self.df = pandas_df

    def query(self, query: str, verify: bool = True) -> Any:
        df = self.df
        inputs = {"objective": query, "df_head": df.head(), "stop": "```"}
        llm_response = LLM_CHAIN.run(**inputs)
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
            return(eval(llm_response))
