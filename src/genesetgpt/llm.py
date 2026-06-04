import openai
import warnings
import anthropic
import numpy as np
import pandas as pd
from typing import Union
from tqdm.auto import tqdm
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
    Summarize a single gene based on multiple literature-based functional summaries. 

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
        warnings.warn(message='The argument prompt_system is set to None, so a general system prompt will be passed to the LLM.')
        prompt_system = "You are an experienced computational biologist with advanced knowledge of analyses such as GWAS, bulk and single cell RNA-seq, spatial 'omics, etc. When generating responses, you consider the statistical, computational, and biological angles of the question at hand. Your responses are detailed without being too overly technical. In all responses, do not utilize any means of referring to a gene other than its HGNC symbol, and do not include any information that is not explicitly present in the user prompt."
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

def summarize_individual_genes(user_prompt_df: pd.DataFrame, 
                               provider: str = 'anthropic', 
                               client: Union[anthropic.Anthropic, openai.OpenAI] = None, 
                               model: str = 'claude-haiku-4-5', 
                               prompt_system: str = None, 
                               n_max_tokens: int = 500, 
                               n_workers: int = 4, 
                               progress_bar: bool = True) -> pd.DataFrame:
    """
    Summarize a set of individual genes in parallel based on their unique, literature-based user prompts as constructed with ``build_prompt_df()``.

    Parameters
    ----------
    user_prompt_df : ``pd.DataFrame``
        A ``pd.DataFrame`` containing a column named 'prompt_user' with the user prompts for each gene as constructed by the ``build_prompt_df()`` function.
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
    progress_bar : ``bool``
        A Boolean specifying whether to display a progress bar during per-gene summarization. Recommended for interactive notebook usage, but should probably be set to False for script usage. Defaults to True.

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
        You are an experienced computational biologist with extensive knowledge of next-generation sequencing analyses such as GWAS, bulk and single cell RNA-seq, spatial 'omics, etc. When generating responses, you consider the statistical, computational, and biological angles of the question at hand. Your responses are detailed without being too overly technical. In all responses, do not utilize any means of referring to a gene other than its HGNC symbol i.e., never use 'Neurogranin' to refer to the gene NRGN. Lastly, do not include any information that is not explicitly present in the user prompt.
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
        iterator = pool.map(summarize_all, user_prompts)
        if progress_bar:
            results = list(
                tqdm(
                    iterator, 
                    total=len(user_prompts), 
                    desc='Generating gene-level LLM summaries'
                )
            )
        else:
            results = list(iterator)
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

class GeneSetSummaryLaTeX(BaseModel):
    formatted_text: str

def summarize_module(module_genes: list, 
                     gene_sumy_df: pd.DataFrame, 
                     prompt_system: str = None, 
                     provider: str = 'anthropic', 
                     client: Union[anthropic.Anthropic, openai.OpenAI] = None, 
                     model: str = 'claude-haiku-4-5', 
                     n_max_tokens: int = 1500, 
                     add_latex_formatted_sumy: bool = False) -> dict[pd.DataFrame, str]:
    """
    Summarize a gene module based on previously-generated, unique LLM summaries of each gene in the module. 

    Parameters
    ----------
    module_genes: ``list``
        A list of strings specifying the HGNC symbols of the genes in the module of interest.
    gene_sumy_df : ``pd.DataFrame``
        A ``pd.DataFrame`` containing the previously-generated LLM summaries of each individual gene.
    prompt_system : ``str``
        A string defining the system prompt that specifies the desired LLM role, along with additional biological context.
    provider : ``str``
        A string specifying the backend LLM provider to use. Must be one of 'anthropic' or 'openai'. Defaults to 'anthropic'.
    client : ``Union[anthropic.Anthropic, openai.OpenAI]``
        An object of class ``Anthropic`` or ``OpenAI`` generated with your API key. Defaults to None. 
    model : ``str``
        A string specifying the specific LLM to use when generating the response. Defaults to 'claude-haiku-4-5'. 
    n_max_tokens : ``int``
        An integer specifying the maximum number of output tokens used by the LLM when summarizing the gene module. Defaults to 1500.
    add_latex_formatted_sumy : ``bool``
        A Boolean specifying whether to add additional columns containing LaTeX-formatted versions of the gene module summary and confidence score rationale to the generated ``pd.DataFrame`` gene module summary in the returned dictionary object. This is mainly useful for manuscript preparation. Defaults to False.
    
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
        warnings.warn(message='The argument prompt_system is set to None; so a less-detailed system prompt will be passed to the LLM.')
        prompt_system = "You are an experienced computational biologist with extensive knowledge of analyses such as GWAS, bulk and single cell RNA-seq, spatial 'omics, etc. When generating responses, you consider the statistical, computational, and biological angles of the question at hand. Your responses are detailed without being too overly technical. In all responses, do not utilize any means of referring to a gene other than its HGNC symbol i.e., never use 'Neurogranin' to refer to the gene NRGN. Lastly, do not include any information that is not explicitly present in the user prompt."
    mask = gene_sumy_df['hgnc_symbol'].isin(values=module_genes)
    module_gene_ids = gene_sumy_df[mask].copy()
    module_llm_summaries_bulleted = '\n'.join(
        f'- **{gene}**: {summary}'
        for gene, summary in zip(module_gene_ids['hgnc_symbol'], module_gene_ids['llm_summary'])
    )
    summary_prompt = f"""
    # Individual Gene Summaries
    Below are independently-generated descriptions for each gene in a module:
    
    <gene_descriptions>
    {module_llm_summaries_bulleted}
    </gene_descriptions>

    # Instructions
    Analyze the functional descriptions provided above and synthesize an annotation for this gene set. Your response must fulfill the following criteria:

    1. **Shared Function Summary**: Write a concise (5–7 sentences) paragraph summarizing the shared biological function(s) or pathway(s) of this gene set.
    2. **Confidence Score**: Provide a robust, 3-decimal score ranging from 0 to 1 estimating your overall confidence in this annotation.
    3. **Confidence Score Rationale**: Accompany your confidence score with a short (2-4 sentences) rationale justifying it. Do not hesitate to express and quantify uncertainty if the genes have highly diverse or unclear functions.
    4. **Gene Set Name**: Provide a distinctive 2-5 word name for the gene set based on your annotation.
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
    model_json = parsed_data.model_dump_json()
    summary_df = pd.DataFrame(
        data={
            'summary': parsed_data.summary, 
            'name': parsed_data.name, 
            'score': parsed_data.confidence_score, 
            'score_rationale': parsed_data.confidence_score_rationale
        }, 
        index=[0]
    )
    if add_latex_formatted_sumy:
        n_max_tokens_latex = max(500, n_max_tokens // 3)
        latex_system_prompt = 'You are a precise text-formatting assistant. Your single task is to identify human gene symbols (HGNC symbols) within a text block and wrap them in LaTeX italicization formatting for usage in a manuscript.'
        def format_latex(text_to_format: str) -> str:
            format_prompt = f"""
            Please take the text block below, identify any HGNC symbols (e.g., STAT3, EGFR, IL6), 
            and wrap them in standard LaTeX italicization formatting: `\\textit{{GENE_SYMBOL}}`.
            
            Text to format:
            {text_to_format}
            
            Strict Constraints:
            1. Only format legitimate gene symbols, pseudogenes, etc. Do not format standard English words or celltype names. If a backslash separates two closely-related genes e.g., S100A4/A5, do not format the backslash as italicized text, but do italicize the symbols on either side of the backslash.
            2. Under no circumstances may you alter, add to, reword, or delete any other surrounding text or punctuation.
            """
            if provider == 'anthropic':
                res = client.messages.parse(
                    model=model,
                    max_tokens=n_max_tokens_latex,
                    system=latex_system_prompt,
                    messages=[{'role': 'user', 'content': format_prompt}],
                    output_format=GeneSetSummaryLaTeX
                )
                return res.parsed_output.formatted_text
            elif provider == 'openai':
                res = client.responses.parse(
                    model=model,
                    max_output_tokens=n_max_tokens_latex,
                    input=[
                        {'role': 'developer', 'content': latex_system_prompt},
                        {'role': 'user', 'content': format_prompt}
                    ],
                    text_format=GeneSetSummaryLaTeX
                )
                return res.output_parsed.formatted_text
        summary_df['summary_latex'] = format_latex(text_to_format=parsed_data.summary)
        summary_df['score_rationale_latex'] = format_latex(text_to_format=parsed_data.confidence_score_rationale)
    res = {
        'module_summary_df': summary_df, 
        'model_json': model_json
    }
    return res

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
    provider = provider.lower()
    if provider == 'openai':
        resp = client.embeddings.create(input=text, model=embedding_model)
        embed = np.array(resp.data[0].embedding, dtype=np.float32)
    else:
        raise ValueError("Provider currently must be set to 'openai' for embedding generation.")
    return embed
