import openai
import numpy as np
import pandas as pd
from functools import partial
from pydantic import BaseModel
from concurrent.futures import ThreadPoolExecutor

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

def get_embedding(text: str, 
                  openai_client: openai.OpenAI, 
                  embedding_model: str = 'text-embedding-ada-002') -> np.ndarray:
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
    resp = openai_client.embeddings.create(input=text, model=embedding_model)
    embed = np.array(resp.data[0].embedding, dtype=np.float32)
    return embed

def summarize_individual_genes(df_with_prompts: pd.DataFrame, 
                               openai_client: openai.OpenAI, 
                               openai_model: str = 'gpt-5-mini', 
                               prompt_dev: str = None) -> pd.DataFrame:
    summarize_one = partial(
        summarize_genes,
        prompt_dev=prompt_dev,
        openai_client=openai_client,
        openai_model=openai_model
    )
    user_prompts = df_with_prompts['prompt_user'].to_list()
    with ThreadPoolExecutor(max_workers=4) as pool:
        results = list(pool.map(summarize_one, user_prompts))
    llm_summaries, llm_scores = zip(*results)
    df_with_prompts['llm_summary'] = llm_summaries
    df_with_prompts['llm_confidence_score'] = llm_scores
    return df_with_prompts

class GeneSetSummary(BaseModel):
    summary: str
    name: str
    confidence: float

def summarize_module(module_genes: list, 
                     df_with_gene_sumys: pd.DataFrame, 
                     openai_client: openai.OpenAI, 
                     prompt_dev: str = None) -> dict:
    mask = df_with_gene_sumys['hgnc_symbol'].isin(values=module_genes)
    module_gene_ids = df_with_gene_sumys[mask].copy()
    module_user_prompts = module_gene_ids['prompt_user'].to_list()
    module_llm_summaries_bulleted = '\n'.join(f'- {s}' for s in module_user_prompts)
    summary_prompt = f"""
    Below are brief, independent descriptions of genes in a set:

    {module_llm_summaries_bulleted}

    Please write a concise (5–7 sentence) paragraph summarizing the common function(s) of this gene set. In addition, please provide a robust, 3-decimal score ranging from 0-1 estimating how confident you are in your overall annotation. Do not be reluctant to express uncertainty if it appears that the genes in the set have diverse or unclear functions. Lastly, provide a short 2-5 word name for the gene set based on your annotation.
    """
    summary_response = openai_client.responses.parse(
        model='gpt-5-mini', 
        input=[
            {'role': 'developer', 'content': prompt_dev}, 
            {'role': 'user', 'content': summary_prompt}
        ], 
        text_format=GeneSetSummary
    )
    summary_df = pd.DataFrame({
        'summary': summary_response.output_parsed.summary, 
        'name': summary_response.output_parsed.name, 
        'score': summary_response.output_parsed.confidence
    })
    model_json = summary_response.model_dump_json()
    res = {
        'module_summary_df': summary_df, 
        'model_json': model_json
    }
    return res
