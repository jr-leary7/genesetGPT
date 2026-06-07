import time
import random
from curl_cffi import requests
from typing import TypedDict, Optional

class HPAData(TypedDict):
    ensembl_id: str
    go_bp_terms: Optional[list[str]]
    diseases: Optional[list[str]]
    metadata: Optional[str]

def fetch_HPA_data(ensembl_id: str) -> HPAData:
    """
    Fetch a gene's related GO:BP terms & disease involvements from the `Human Protein Atlas`_. 

    Parameters
    ----------
    ensembl_id : ``str``
        A string specifying the Ensembl ID for which data will be scraped. 

    Returns
    -------
        A dictionary containing the Ensembl ID, lists of GO:BP terms and related diseases, and other metadata. 

    .. _Human Protein Atlas: https://www.proteinatlas.org
    """
    hpa_url = f'https://www.proteinatlas.org/{ensembl_id}.json'
    headers = {
        'Referer': 'https://www.proteinatlas.org/',
        'Accept': 'application/json, text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8'
    }
    delay = random.uniform(0.1, 0.25)
    time.sleep(delay)
    try:
        hpa_page = requests.get(
            url=hpa_url,
            impersonate='chrome',
            headers=headers,
            timeout=15
        )
        status_code = hpa_page.status_code
    except Exception as e:
        return {
            'ensembl_id': ensembl_id, 
            'go_bp_terms': None, 
            'diseases': None, 
            'metadata': f'HPA Network/CFFI Error: {str(e)}'
        }
    if status_code != 200:
        res = {
            'ensembl_id': ensembl_id, 
            'go_bp_terms': None, 
            'diseases': None, 
            'metadata': f'Status code {status_code}'
        }
    else:
        hpa_json = hpa_page.json()
        if hpa_json['Biological process'] is None:
            gene_processes = None
        else:
            gene_processes = str.lower(', '.join(hpa_json['Biological process']))
        if hpa_json['Disease involvement'] is None: 
            gene_diseases = None
        else:
            gene_diseases = str.lower(', '.join(hpa_json['Disease involvement']))
        res = {
            'ensembl_id': ensembl_id, 
            'go_bp_terms': gene_processes, 
            'diseases': gene_diseases, 
            'metadata': None
        }
    return res
