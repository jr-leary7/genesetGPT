import requests
from typing import TypedDict, Optional

class HPAData(TypedDict):
    ensembl_id: str
    go_bp_terms: Optional[list[str]]
    diseases: Optional[list[str]]
    metadata: Optional[str]

def fetch_HPA_data(ensembl_id: str) -> HPAData:
    """
    Fetch gene-level data from the Human Protein Atlas. 

    Parameters
    ----------
    ensembl_id : str
        A string specifying the Ensembl ID for which data will be scraped. 

    Returns
    -------
    res: dict 
        A dict containing the Ensembl ID, lists of GO:BP terms and related diseases, and other metadata. 
    """
    hpa_url = 'https://www.proteinatlas.org/' + ensembl_id + '.json'
    hpa_page = requests.get(url=hpa_url)
    status_code = hpa_page.status_code
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
