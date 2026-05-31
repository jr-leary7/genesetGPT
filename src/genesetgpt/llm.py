import openai
import warnings
import anthropic
import numpy as np
import pandas as pd
from typing import Union
from functools import partial
from pydantic import BaseModel
from concurrent.futures import ThreadPoolExecutor
from tenacity import retry, wait_exponential, stop_after_attempt, retry_if_exception_type

class GeneSummary(BaseModel):
    summary: str
    confidence_score: float
    confidence_score_rationale: str

@retry(
    retry=retry_if_exception_type(exception_types=(anthropic.RateLimitError, anthropic.APIConnectionError, openai.RateLimitError)),
    wait=wait_exponential(multiplier=1, min=2, max=10),
    stop=stop_after_attempt(max_attempt_number=5)
)
def summarize_gene(prompt_user: str, 
                   prompt_system: str, 
                   provider: str = 'anthropic', 
                   client: Union[anthropic.Anthropic, openai.OpenAI] = None, 
                   model: str = 'claude-haiku-4-5', 
                   n_max_tokens: int = 500) -> tuple[str, float, str]:
    """
    Summarize a gene given multiple literature-based functional summaries. 

    Parameters
    ----------
    prompt_user : ``str``
        A string containing the previously-generated user prompt.
    prompt_system : ``str``
        A string containing the system prompt.
    provider : ``str``
        A string specifying the backend LLM provider to use. Must be one of 'anthropic' or 'openai'. Defaults to 'anthropic'.
    client : ``Union[anthropic.Anthropic, openai.OpenAI]``
        An object of class ``Anthropic`` or ``OpenAI`` generated with your API key. Defaults to None. 
    model : ``str``
        A string specifying the specific LLM version to use when generating the response. Defaults to 'claude-haiku-4-5'. 
    n_max_tokens : ``int``
        An integer specifying the maximum number of output tokens used by the LLM when summarizing the gene. Defaults to 500.

    Returns
    -------
        A tuple containing the gene-level summary, estimated confidence score, and confidence score rationale. 
    """
    provider = provider.lower()
    if provider not in ['anthropic', 'openai']:
        raise ValueError("Provider must be one of 'anthropic' or 'openai'.")
    if client is None:
        raise ValueError('A client object generated with your API key must be passed to enable any LLM usage.')
    if prompt_system is None:
        warnings.warn(message='The argument prompt_system is set as None; so a less-detailed system prompt will be passed to the LLM.')
        prompt_system = """
        You are an experienced computational biologist with advanced knowledge of analyses such as GWAS, bulk and single cell RNA-seq, spatial 'omics, etc. When generating responses, you consider the statistical, computational, and biological angles of the question at hand. Your responses are detailed without being too overly technical.
        """
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
        res_tuple = (
            llm_res.parsed_output.summary, 
            llm_res.parsed_output.confidence_score, 
            llm_res.parsed_output.confidence_score_rationale
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
    text : ``str``
        A string containing the text to be embedded.
    provider : ``str``
        A string specifying the backend LLM provider to use. Must be equal to 'openai', as Anthropic currently does not natively support embedding models.
    client : ``openai.OpenAI``
        An object of class ``OpenAI`` generated with your API key.
    embedding_model : ``str``
        A string specifying the embedding model to be used. Defaults to 'text-embedding-ada-002'. 

    Returns
    -------
    embed : ``np.array``
        The LLM-generated embedding. 
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
                               n_max_tokens: int = 500, 
                               n_workers: int = 4) -> pd.DataFrame:
    """
    Summarize individual genes in parallel based on their unique, literature-based user prompts as constructed with ``build_prompt_df()``.

    Parameters
    ----------
    user_prompt_df : ``pd.DataFrame``
        A ``pd.DataFrame`` containing a column named 'prompt_user' with the user prompts for each gene as generated by the ``build_prompt_df()`` function.
    provider : ``str``
        A string specifying the backend LLM provider to use. Must be one of 'anthropic' or 'openai'. Defaults to 'anthropic'.
    client : ``Union[anthropic.Anthropic, openai.OpenAI]``
        An object of class ``Anthropic`` or ``OpenAI`` generated with your API key. Defaults to None. 
    model : ``str``
        A string specifying the specific LLM version to use when generating each response. Defaults to 'claude-haiku-4-5'. 
    prompt_system : ``str``
        A string containing the system prompt specifying the LLM's role and additional biological context. Defaults to None.
    n_max_tokens : ``int``
        An integer specifying the maximum number of output tokens used by the LLM when summarizing the gene. Defaults to 500.
    n_workers : ``int``
        An integer specifying the number of workers to use for parallel processing. Defaults to 4.

    Returns
    -------
    user_prompt_df : ``pd.DataFrame``
        The inputted ``pd.DataFrame`` with three additional columns containing each gene's LLM-generated functional summary, estimated confidence score, and confidence score rationale.
    """
    provider = provider.lower()
    if provider not in ['anthropic', 'openai']:
        raise ValueError("Provider must be one of 'anthropic' or 'openai'.")
    if client is None:
        raise ValueError('A client object initialized with your API key must be passed to enable any LLM usage.')
    if prompt_system is None:
        warnings.warn(message='The argument prompt_system is set as None; so a less-detailed system prompt will be passed to the LLM.')
        prompt_system = """
        You are an experienced computational biologist with advanced knowledge of analyses such as GWAS, bulk and single cell RNA-seq, spatial 'omics, etc. When generating responses, you consider the statistical, computational, and biological angles of the question at hand. Your responses are detailed without being too overly technical.
        """
    summarize_all = partial(
        summarize_gene,
        prompt_system=prompt_system,
        provider=provider,
        client=client,
        model=model, 
        n_max_tokens=n_max_tokens
    )
    user_prompts = user_prompt_df['prompt_user'].to_list()
    with ThreadPoolExecutor(max_workers=n_workers) as pool:
        results = list(pool.map(summarize_all, user_prompts))
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
                     n_max_tokens: int = 1500) -> dict[pd.DataFrame, str]:
    """
    Summarize a gene module based on previously-generated LLM sumaries of each individual gene belonging to said module. 

    Parameters
    ----------
    module_genes: ``list``
        A list of strings specifying the HGNC symbols of the genes in the module of interest.
    gene_sumy_df : ``pd.DataFrame``
        A ``pd.DataFrame`` containing the previously-generated LLM summaries of each individual gene.
    prompt_system : ``str``
        A string containing the system prompt that specifies the desired LLM role, along with additional biological context.
    provider : ``str``
        A string specifying the backend LLM provider to use. Must be one of 'anthropic' or 'openai'. Defaults to 'anthropic'.
    client : ``Union[anthropic.Anthropic, openai.OpenAI]``
        An object of class ``Anthropic`` or ``OpenAI`` generated with your API key. Defaults to None. 
    model : ``str``
        A string specifying the specific LLM to use when generating the response. Defaults to 'claude-haiku-4-5'. 
    n_max_tokens : ``int``
        An integer specifying the maximum number of output tokens used by the LLM when summarizing the gene module. Defaults to 1500.
    
    Returns
    -------
        A dictionary containing the gene module summary / name / confidence score / confidence score rationale formatted as a ``pd.DataFrame``, along with the raw LLM response's JSON output formatted as a string. 
    """
    provider = provider.lower()
    if provider not in ['anthropic', 'openai']:
        raise ValueError("Provider must be one of 'anthropic' or 'openai'.")
    if client is None:
        raise ValueError('A client object generated with your API key must be passed to enable any LLM usage.')
    if prompt_system is None:
        warnings.warn(message='The argument prompt_system is set as None; so a less-detailed system prompt will be passed to the LLM.')
        prompt_system = """
        You are an experienced computational biologist with advanced knowledge of analyses such as GWAS, bulk and single cell RNA-seq, spatial 'omics, etc. When generating responses, you consider the statistical, computational, and biological angles of the question at hand. Your responses are detailed without being too overly technical.
        """
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
    summary_df = pd.DataFrame(data={
        'summary': parsed_data.summary, 
        'name': parsed_data.name, 
        'score': parsed_data.confidence_score, 
        'score_rationale': parsed_data.confidence_score_rationale
    })
    model_json = parsed_data.model_dump_json()
    res = {
        'module_summary_df': summary_df, 
        'model_json': model_json
    }
    return res
