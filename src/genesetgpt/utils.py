import requests
import xmltodict
import numpy as np
from typing import TypedDict, Optional

def add_trailing_period(text: str) -> str:
    """
    Add a final period to a string. 

    Parameters
    ----------
    text : str
        A string to be edited. Defaults to None. 

    Returns
    -------
    text : str
        A string with a trailing period added if needed. 
    """
    if not text.endswith('.'):
        return text + '.'
    return text

def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    """
    Compute the cosine similarity between two `numpy` arrays. 

    Parameters
    ----------
    a : np.ndarray
        A `numpy` array. 
    b : np.ndarray
        A `numpy` array of the same dimension as `a`. 

    Returns
    -------
    res : float 
        A float specifying the cosine similarity between `a` and `b`. 
    """
    res = float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))
    return res 

class GeneAliases(TypedDict):
    hgnc_symbol: str
    aliases: Optional[list[str]]
    metadata: Optional[str]

def get_aliases(hgnc_symbol: str) -> GeneAliases:
    """
    Fetch the known aliases for a given HGNC symbol. 

    Parameters
    ----------
    hgnc_symbol : str
        A string specifying the HGNC symbol of the gene of interest.
    
    Returns
    -------
    res : dict 
        A dict containing the HGNC symbol, a list of any known aliases / previous symbols, and associated metadata. 
    """
    gene_url = f'https://rest.genenames.org/fetch/symbol/{hgnc_symbol}'
    gene_page = requests.get(url=gene_url)
    status_code = gene_page.status_code
    if status_code != 200:
        res = {
            'hgnc_symbol': hgnc_symbol, 
            'aliases': None, 
            'metadata': f'Status code {status_code}'
        }
    else: 
        gene_xml = xmltodict.parse(xml_input=gene_page.text)
        arr_list = (
            gene_xml
            .get('response', {})
            .get('result', {})
            .get('doc', {})
            .get('arr', [])
        )
        if not isinstance(arr_list, list):
            gene_aliases = set()
        else:
            alias_entry = next(
                (entry for entry in arr_list if entry.get('@name') == 'alias_symbol'),
                {}
            )
            gene_aliases = alias_entry.get('str', [])
            if isinstance(gene_aliases, str):
                gene_aliases = [gene_aliases]
            if not isinstance(gene_aliases, list):
                gene_aliases = set()
        res = {
            'hgnc_symbol': hgnc_symbol, 
            'aliases': list(gene_aliases), 
            'metadata': None
        }
    return res
