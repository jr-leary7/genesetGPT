import openai
import numpy as np
from pydantic import BaseModel

class GeneSummary(BaseModel):
    summary: str
    confidence: float

def summarize_genes(prompt_user: str, 
                    prompt_dev: str, 
                    openai_client: openai.OpenAI, 
                    openai_model: str = 'gpt-5-mini') -> tuple[str, float]:
    """
    Summarize a gene given several functional summaries. 

    Parameters
    ----------
    prompt_user : str
        A string containing the user prompt.
    prompt_dev : str
        A string containing the developer prompt.
    openai_client : openai.OpenAI
        An object of class `OpenAI`. 
    openai_model : str
        A string specifying the OpenAI model to use when generating the response. Defaults to 'gpt-5-mini'. 

    Returns
    -------
        A tuple containing the gene-level summary and associated confidence score. 
    """
    res = openai_client.responses.parse(
        model=openai_model,
        input=[
            {'role': 'developer', 'content': prompt_dev},
            {'role': 'user', 'content': prompt_user},
        ],
        text_format=GeneSummary,
    )
    return res.output_parsed.summary, res.output_parsed.confidence

def get_embedding(text: str, embedding_model: str = 'text-embedding-ada-002') -> np.ndarray:
    """
    Generate an embedding for a given string of text. 

    Parameters
    ----------
    text : str
        A string containing the text to be embedded.
    embedding_model : str
        A string specifying the embedding model to be used. Defaults to 'text-embedding-ada-002'. 

    Returns
    -------
    embed : np.array
        A `numpy` array containing the embedding. 
    """
    resp = openai.embeddings.create(input=text, model=embedding_model)
    embed = np.array(resp.data[0].embedding, dtype=np.float32)
    return embed
