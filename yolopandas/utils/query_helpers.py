from typing import Any

from langchain.callbacks import get_openai_callback

from yolopandas import pd


def run_query_with_cost(df: pd.DataFrame, query: str, yolo: bool = False) -> Any:
    """
    A function to run a YOLOPandas query with cost estimation returned for your query in terms of tokens used.
    This includes total tokens, prompt tokens, completion tokens, and the total cost in USD.

    Parameters
    ----------
    df : pd.DataFrame
        The Pandas DataFrame with your data
    query : str
        The query you want to run against your data
    yolo : bool
        Boolean value used to return a prompt to a user or not to accept the code result before
        running the code (False means to return the prompt)

    Returns
    -------
    result : Any
        The results of the query run against your data. A prompt may be returned as intermediary
        output to proceed with generating the result or not.
    """
    with get_openai_callback() as cb:
        result = df.llm.query(query, yolo=yolo)
        print(f"Total Tokens: {cb.total_tokens}")
        print(f"Prompt Tokens: {cb.prompt_tokens}")
        print(f"Completion Tokens: {cb.completion_tokens}")
        print(f"Total Cost (USD): ${cb.total_cost}")
        return result
