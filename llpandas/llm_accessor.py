import pandas as pd

from llpandas.chains import LLM_CHAIN


@pd.api.extensions.register_dataframe_accessor("llm")
class LLMAccessor:

    def __init__(self, pandas_df: pd.DataFrame):
        self.df = pandas_df

    def query(self, query):
        df = self.df
        inputs = {"objective": query, "df_head": df.head(), "stop": "```"}
        llm_response = LLM_CHAIN.run(**inputs)
        print("suggested code:")
        print(llm_response)
        print("run this code? y/n")
        user_input = input()
        if user_input == "y":
            return(eval(llm_response))
