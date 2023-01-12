import pandas as pd

from llpandas.chains import LLM_CHAIN


class LLDF(pd.DataFrame):

    @property
    def _constructor(self):
        return LLDF

    def llm(self, query):
        inputs = {"objective": query, "df_head": self.head(), "stop": "```"}
        llm_response = LLM_CHAIN.run(**inputs)
        print("suggested code:")
        print(llm_response)
        print("run this code? y/n")
        user_input = input()
        if user_input == "y":
            return(eval(llm_response))
