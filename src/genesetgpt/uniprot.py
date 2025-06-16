import re
import time 
import requests
from typing import TypedDict, Optional
from unipressed import IdMappingClient
from .utils import add_trailing_period 

class CanonicalProteinProduct(TypedDict):
    ensembl_id: str
    canonical_protein_product: str 

def fetch_canonical_protein_product(ensembl_id: str, sleep_interval: float = 1.0) -> CanonicalProteinProduct:
    """
    Identify the canonical protein product of a given gene. 

    Parameters
    ----------
    ensembl_id : str 
        A string specifying the Ensembl ID of the gene of interest. Defaults to None.
    sleep_interval : float
        A float specifying how long the UniProt request should be left to wait before fetching results. Defaults to 1. 

    Returns
    -------
    res : dict
        A dict containing the gene's Ensembl ID and the UniProt ID of its canonical protein product. 
    """
    request = IdMappingClient.submit(
        source='Ensembl', 
        dest='UniProtKB', 
        ids=[ensembl_id]
    )
    time.sleep(sleep_interval)
    res = list(request.each_result())
    if len(res) == 0:
        res = {
            'ensembl_id': ensembl_id, 
            'canonical_protein_product': None
        }
    else:
        canonical_uniprot_id = list(request.each_result())[0]['to']
        res = {
            'ensembl_id': ensembl_id, 
            'canonical_protein_product': canonical_uniprot_id
        }
    return res

def clean_uniprot_summary(text: str) -> str:
    """
    Clean up the UniProt summary. 

    Parameters
    ----------
    text : str 
        A string containing the UniProt summary. Defaults to None. 

    Returns
    -------
    text_clean : str
        A string containing the reformatted summary. 
    """
    text_clean = re.sub(r'\(\s*PubMed:\d+(?:,\s*PubMed:\d+)*\s*\)', '', text)
    text_clean = re.sub(r'\s{2,}', ' ', text_clean)
    text_clean = re.sub(r'\s*,\s*,', ',', text_clean)
    text_clean = re.sub(r'\s*,\s*$', '', text_clean)
    text_clean = re.sub(r'\s*;\s*$', '', text_clean)
    text_clean = text_clean.strip()
    return text_clean

class UniProtSummary(TypedDict):
    ensembl_id: str
    uniprot_id: str
    uniprot_functions: Optional[list[str]]
    metadata: Optional[str]

def fetch_uniprot_summary(ensembl_id: str) -> UniProtSummary:
    """
    Fetch the UniProt summary of the canonical protein product of a given gene. 

    Parameters
    ----------
    ensembl_id : str 
        A string specifying the Ensembl ID of the gene of interest. Defaults to None.

    Returns
    -------
    res : dict 
        A dict containing (if the request is successful) the Ensembl ID, corresponding UniProt ID, a list of UniProt functional summaries, and associated metadata. 
    """
    uniprot_id = fetch_canonical_protein_product(ensembl_id=ensembl_id)['canonical_protein_product']
    if uniprot_id is None:
        res = {
            'ensembl_id': ensembl_id, 
            'uniprot_id': None, 
            'uniprot_functions': None, 
            'metadata': None
        }
    else:
        uniprot_url = 'https://rest.uniprot.org/uniprotkb/search?&query=accession:'
        uniprot_url += uniprot_id
        uniprot_url += '&organism_id=9606&format=json'
        uniprot_page = requests.get(uniprot_url)
        status_code = uniprot_page.status_code
        if status_code != 200:
            res = {
                'ensembl_id': ensembl_id, 
                'uniprot_id': uniprot_id, 
                'uniprot_functions': None, 
                'metadata': f'Status code {status_code}'
            }
        else:
            uniprot_json = uniprot_page.json()
            if 'comments' in uniprot_json['results'][0].keys():
                uniprot_comments = uniprot_json['results'][0]['comments']
                uniprot_functions = []
                for elem in uniprot_comments:
                    comment_type = elem.get('commentType')
                    if comment_type == 'FUNCTION':
                        function_text = elem['texts'][0]['value']
                        function_text = clean_uniprot_summary(text=function_text)
                        function_text = add_trailing_period(text=function_text)
                        uniprot_functions.append(function_text)
                if len(uniprot_functions) == 0:
                    uniprot_functions = None
                res = {
                    'ensembl_id': ensembl_id, 
                    'uniprot_id': uniprot_id, 
                    'uniprot_functions': uniprot_functions, 
                    'metadata': None
                }
            else:
                res = {
                    'ensembl_id': ensembl_id, 
                    'uniprot_id': uniprot_id, 
                    'uniprot_functions': None, 
                    'metadata': None
                }
    return res
