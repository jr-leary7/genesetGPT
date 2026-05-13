import openai
import pandarallel
import pandas as pd
from .hpa import fetch_HPA_data
from .mim import fetch_mim_summary 
from .entrez import fetch_entrez_summary 
from .uniprot import fetch_uniprot_summary
from .utils import add_trailing_period, get_aliases

def build_user_prompt(ensembl_id: str, 
                      hgnc_symbol: str, 
                      entrez_id: str, 
                      entrez_email: str, 
                      mim_mapping_table: pd.DataFrame, 
                      mim_api_key: str, 
                      include_aliases: bool = True) -> str: 
    """
    Generate a user prompt for a given gene. 

    Parameters
    ----------
    ensembl_id : str
        A string specifying the Ensembl ID of the gene of interest.
    hgnc_symbol : str
        A string specifying the HGNC symbol of the gene of interest.
    entrez_id : str
        A string specifying the Entrez ID of the gene of interest.
    entrez_email : str
        A string specifying the email address associated with the Entrez query.
    mim_mapping_table : pd.DataFrame
        A DataFrame containing the mapping from MIM ID to Ensembl ID. 
    mim_api_key : str
        A string specifying the API key for the MIM database.

    Returns
    -------
    prompt_user : str 
        A (long) string containing the generated user prompt. 

    """
    if include_aliases: 
        gene_aliases = get_aliases(hgnc_symbol=hgnc_symbol)
        if gene_aliases['aliases'] is not None:
            gene_aliases = ', '.join(gene_aliases['aliases'])
        else: 
            gene_aliases = None
    hpa_data = fetch_HPA_data(ensembl_id=ensembl_id)
    go_bp_terms = hpa_data['go_bp_terms']
    diseases = hpa_data['diseases']
    uniprot_info = fetch_uniprot_summary(ensembl_id=ensembl_id)
    if uniprot_info['uniprot_functions'] is None:
        uniprot_summary = None 
    elif len(uniprot_info['uniprot_functions']) > 1:
        uniprot_summary = ' '.join(uniprot_info['uniprot_functions'])
    else:
        uniprot_summary = uniprot_info['uniprot_functions'][0]
    entrez_summary = fetch_entrez_summary(entrez_id=entrez_id, entrez_email=entrez_email)
    mim_info = fetch_mim_summary(
        ensembl_id=ensembl_id, 
        mapping_table=mim_mapping_table, 
        mim_api_key=mim_api_key
    )
    if mim_info['mim_summary'] is None:
        mim_summary = None
    elif len(mim_info['mim_summary']) > 1:
        mim_summary = ' '.join(mim_info['mim_summary'])
    else: 
        mim_summary = mim_info['mim_summary'][0]
    prompt_user = f'I have collected several functional summaries concerning the human gene {hgnc_symbol}'
    prompt_user += f' (Ensembl ID {ensembl_id}, Entrez ID {entrez_id}). '
    if gene_aliases is not None:
        prompt_user += 'Known aliases for this gene include: '
        prompt_user += gene_aliases
        prompt_user += '. '
    prompt_user += 'Please coalesce the various summaries into a single 3-5 sentence description of the function of the gene. In addition, please provide a score ranging from 0-1 specifying how confident you are in your summarization. '
    if go_bp_terms is not None:
        prompt_user += 'According to the Human Protein Atlas (HPA) the Gene Ontology Biological Process (GO:BP) terms this gene is involved in are: '
        prompt_user += go_bp_terms
    if diseases is not None:
        prompt_user += '. The HPA specifies that the diseases this gene is implicated in are: '
        prompt_user += diseases
    if mim_info['mim_summary'] is not None:
        prompt_user += '. The Mendelian Inheritance of Man (MIM) summary for this gene is: '
        prompt_user += mim_summary
    if uniprot_summary is not None:
        prompt_user += ' The UniProt functional summary for the gene is: '
        prompt_user += uniprot_summary
    if entrez_summary is not None:
        prompt_user += ' The Entrez summary for the gene is: '
        prompt_user += entrez_summary['entrez_summary']
    prompt_user = add_trailing_period(text=prompt_user)
    return prompt_user

def build_prompt_df(gene_list: list, 
                    gene_id_table: pd.DataFrame, 
                    mim_mapping_table: pd.DataFrame, 
                    mim_api_key: str = None, 
                    n_cores: int = 2, 
                    progress_bar: bool = True) -> pd.DataFrame:
    mask = gene_id_table['hgnc_symbol'].isin(values=gene_list)
    gene_id_table = gene_id_table[mask].copy()
    gene_id_table.dropna(inplace=True)
    pandarallel.initialize(
        progress_bar=progress_bar, 
        nb_workers=n_cores, 
        verbose=0
    )
    gene_id_table['prompt_user'] = gene_id_table.parallel_apply(
        lambda row: 
        build_user_prompt(
            ensembl_id=row['ensembl_id'], 
            hgnc_symbol=row['hgnc_symbol'], 
            entrez_id=row['entrez_id'], 
            entrez_email='j.leary@ufl.edu', 
            mim_mapping_table=mim_mapping_table, 
            mim_api_key=mim_api_key, 
            include_aliases=True
        ), 
        axis=1
    )
    return gene_id_table
