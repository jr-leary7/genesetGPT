import openai
import anthropic
import numpy as np
import pandas as pd
from typing import Union, Any
from functools import partial
from pydantic import BaseModel
from concurrent.futures import ThreadPoolExecutor
class GeneSummary(BaseModel):
    summary: str
    confidence_score: float
    confidence_score_rationale: str

def summarize_gene(prompt_user: str, 
                   prompt_system: str, 
                   provider: str = 'anthropic', 
                   client: Union[anthropic.Anthropic, openai.OpenAI] = None, 
                   model: str = 'claude-haiku-4-5', 
                   n_max_tokens: int = 2000) -> tuple[str, float, str]:
    """
    Summarize a gene given several literature-based functional summaries. 

    Parameters
    ----------
    prompt_user : str
        A string containing the user prompt.
    prompt_system : str
        A string containing the system prompt.
    provider : str
        A string specifying the backend LLM provider to use. Must be one of 'anthropic' or 'openai'. Defaults to 'anthropic'.
    client : Union[anthropic.Anthropic, openai.OpenAI]
        An object of class `Anthropic` or `OpenAI` generated with your API key. Defaults to None. 
    model : str
        A string specifying the specific LLM to use when generating the response. Defaults to 'claude-haiku-4-5'. 

    Returns
    -------
        A tuple containing the gene-level summary, confidence score, and confidence score rationale. 
    """
    if provider not in ['anthropic', 'openai']:
        raise ValueError("Provider must be one of 'anthropic' or 'openai'.")
    if client is None:
        raise ValueError('A client object generated with an API key must be passed to enable any LLM usage.')
    if provider == 'anthropic':
        llm_res = client.messages.parse(
            model=model, 
            max_tokens=n_max_tokens, 
            system=prompt_system, 
            messages=[
                {'role': 'user', 'content': prompt_user}
            ], 
            output_format=GeneSummary
        )
    elif provider == 'openai':
        llm_res = client.responses.parse(
            model=model,
            max_output_tokens=n_max_tokens,
            input=[
                {'role': 'developer', 'content': prompt_system},
                {'role': 'user', 'content': prompt_user},
            ],
            text_format=GeneSummary
        )
    res_tuple = (
        llm_res.output_parsed.summary, 
        llm_res.output_parsed.confidence_score, 
        llm_res.output_parsed.confidence_score_rationale
    )
    return res_tuple

def get_embedding(text: str, 
                  provider: str, 
                  client: openai.OpenAI, 
                  embedding_model: str = 'text-embedding-ada-002') -> np.ndarray:
    """
    Generate a numerical embedding for a given text string. 

    Parameters
    ----------
    text : str
        A string containing the text to be embedded.
    provider : str
        A string specifying the backend LLM provider to use. Must be one of 'anthropic' or 'openai'.
    client : openai.OpenAI
        An object of class `OpenAI` generated with your API key.
    embedding_model : str
        A string specifying the embedding model to be used. Defaults to 'text-embedding-ada-002'. 

    Returns
    -------
    embed : np.array
        A `numpy` array containing the embedding. 
    """
    if provider == 'openai':
        resp = client.embeddings.create(input=text, model=embedding_model)
        embed = np.array(resp.data[0].embedding, dtype=np.float32)
    return embed

def summarize_individual_genes(user_prompt_df: pd.DataFrame, 
                               provider: str = 'anthropic', 
                               client: Union[anthropic.Anthropic, openai.OpenAI] = None, 
                               model: str = 'claude-haiku-4-5', 
                               prompt_system: str = None, 
                               n_workers: int = 4) -> pd.DataFrame:
    if provider not in ['anthropic', 'openai']:
        raise ValueError("Provider must be one of 'anthropic' or 'openai'.")
    if client is None:
        raise ValueError('A client object generated with an API key must be passed to enable any LLM usage.')
    summarize_one = partial(
        summarize_gene,
        prompt_system=prompt_system,
        provider=provider,
        client=client,
        model=model
    )
    user_prompts = user_prompt_df['prompt_user'].to_list()
    with ThreadPoolExecutor(max_workers=n_workers) as pool:
        results = list(pool.map(summarize_one, user_prompts))
    llm_summaries, llm_scores, llm_score_rationales = zip(*results)
    user_prompt_df['llm_summary'] = llm_summaries
    user_prompt_df['llm_confidence_score'] = llm_scores
    user_prompt_df['llm_confidence_score_rationale'] = llm_score_rationales
    return user_prompt_df

class GeneSetSummary(BaseModel):
    summary: str
    name: str
    confidence_score: float
    confidence_score_rationale: str

def summarize_module(module_genes: list, 
                     gene_sumy_df: pd.DataFrame, 
                     prompt_system: str = None, 
                     provider: str = 'anthropic', 
                     client: Union[anthropic.Anthropic, openai.OpenAI] = None, 
                     model: str = 'claude-haiku-4-5', 
                     n_max_tokens: int = 6000) -> dict[str, Any]:
    """
    Summarize a gene module based on previously-generated LLM sumaries of individual genes. 

    Parameters
    ----------
    module_genes: list
        A list of strings specifying the HGNC symbols of the genes in the module of interest.
    gene_sumy_df : pd.DataFrame
        A DataFrame containing the previously-generated LLM summaries of each individual gene.
    prompt_system : str
        A string containing the system prompt specifying the LLM's role and additional biological context.
    provider : str
        A string specifying the backend LLM provider to use. Must be one of 'anthropic' or 'openai'. Defaults to 'anthropic'.
    client : Union[anthropic.Anthropic, openai.OpenAI]
        An object of class `Anthropic` or `OpenAI` generated with your API key. Defaults to None. 
    model : str
        A string specifying the specific LLM to use when generating the response. Defaults to 'claude-haiku-4-5'. 
    n_max_tokens : int
        An integer specifying the maximum number of output tokens used by the LLM when summarizing the gene module. Defaults to 6000.
    
    Returns
    -------
        A dict containing the gene module summary dataframe and the raw LLM response JSON. 
    """
    if provider not in ['anthropic', 'openai']:
        raise ValueError("Provider must be one of 'anthropic' or 'openai'.")
    if client is None:
        raise ValueError('A client object generated with an API key must be passed to enable any LLM usage.')
    mask = gene_sumy_df['hgnc_symbol'].isin(values=module_genes)
    module_gene_ids = gene_sumy_df[mask].copy()
    module_user_prompts = module_gene_ids['prompt_user'].to_list()
    module_llm_summaries_bulleted = '\n'.join(f'- {s}' for s in module_user_prompts)
    summary_prompt = f"""
    Below are brief, independent descriptions of genes in a set:

    {module_llm_summaries_bulleted}

    Please write a concise (5–7 sentences) paragraph summarizing the shared function(s) of this gene set. In addition, please provide a robust, 3-decimal score ranging from 0-1 estimating how confident you are in your overall annotation, along with an accompanying short rationale justifying your confidence score. Do not be reluctant to express and quantify uncertainty if it appears that the genes in the set have diverse or unclear functions. Lastly, provide a short 2-5 word name for the gene set based on your annotation.
    """
    if provider == 'anthropic':
        summary_response = client.messages.parse(
            model=model, 
            max_tokens=n_max_tokens, 
            system=prompt_system, 
            messages=[
                {'role': 'user', 'content': summary_prompt}
            ], 
            output_format=GeneSetSummary
        )
        parsed_data = summary_response.parsed_output
    elif provider == 'openai':
        summary_response = client.responses.parse(
            model=model, 
            max_output_tokens=n_max_tokens,
            input=[
                {'role': 'developer', 'content': prompt_system}, 
                {'role': 'user', 'content': summary_prompt}
            ], 
            text_format=GeneSetSummary
        )
        parsed_data = summary_response.output_parsed
    summary_df = pd.DataFrame({
        'summary': parsed_data.output_parsed.summary, 
        'name': parsed_data.output_parsed.name, 
        'score': parsed_data.output_parsed.confidence_score, 
        'score_rationale': parsed_data.output_parsed.confidence_score_rationale
    })
    model_json = parsed_data.model_dump_json()
    res = {
        'module_summary_df': summary_df, 
        'model_json': model_json
    }
    return res
