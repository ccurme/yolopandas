from langchain.llms.base import BaseLLM

from llpandas.chains import LLM_CHAIN


def set_llm(llm: BaseLLM) -> None:
    LLM_CHAIN.llm = llm
