from langchain.llms.base import BaseLLM

from llpandas import llm_accessor
from llpandas.chains import LLM_CHAIN


def set_llm(llm: BaseLLM):
    LLM_CHAIN.llm = llm
