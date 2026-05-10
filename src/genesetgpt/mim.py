import re 
import time 
import xmltodict 
import pandas as pd
from io import StringIO
from curl_cffi import requests
from typing import TypedDict, Optional
from .utils import add_trailing_period 

def fetch_mim_table(sort_by: str = 'hgnc_symbol') -> pd.DataFrame:
    """
    Fetch a table containing a mapping of MIM IDs to gene IDs. 

    Parameters
    ----------
    sort_by : str 
        A string specifying the gene ID to sort the table by. Defaults to 'hgnc_symbol'. 

    Returns
    -------
    mim_mapping_df : pd.DataFrame 
        A DataFrame that specifies the relationships between MIM ID, Ensembl ID, Entrez ID, and HGNC symbol. 
    """
    headers = {
        'Referer': 'https://www.omim.org/',
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8'
    }
    response = requests.get(
        url='https://www.omim.org/static/omim/data/mim2gene.txt', 
        impersonate='chrome',
        headers=headers
    )
    if response.status_code == 200:
        mim_mapping_df = pd.read_csv(
            filepath_or_buffer=StringIO(response.text), 
            delimiter='\t', 
            skiprows=4
        )
    else: 
        raise ValueError(f'The returned HTTP status code was: {response.status_code}')
    mim_mapping_df.drop_duplicates(inplace=True)
    mim_mapping_df.rename(
        columns={
            '# MIM Number': 'mim_id', 
            'MIM Entry Type (see FAQ 1.3 at https://omim.org/help/faq)': 'mim_entry_type', 
            'Entrez Gene ID (NCBI)': 'entrez_id', 
            'Approved Gene Symbol (HGNC)': 'hgnc_symbol', 
            'Ensembl Gene ID (Ensembl)': 'ensembl_id'
        }, 
        inplace=True
    )
    if sort_by not in mim_mapping_df.columns: 
        raise KeyError(f"`sort_by='{sort_by}'` is not a valid column.")
    mim_mapping_df.sort_values(by=sort_by, inplace=True)
    return mim_mapping_df 

class MIMSummary(TypedDict):
    ensembl_id: str
    mim_ids: Optional[list[str]]
    mim_summary: Optional[list[str]]

def fetch_mim_summary(ensembl_id: str, 
                      mapping_table: pd.DataFrame, 
                      mim_api_key: str, 
                      sleep_interval: float = 1.0) -> MIMSummary:
    """
    Fetch the MIM summary for a given gene of interest. 

    Parameters
    ----------
    ensembl_id : str 
        A string specifying the Ensembl ID of the gene of interest.
    mapping_table : pd.DataFrame 
        A DataFrame containing the mapping from MIM ID to Ensembl ID. 
    mim_api_key : str 
        A string specifying the API key for the MIM database. 
    sleep_interval : float
        A float specifying how long the MIM request should be left to wait before fetching results. Defaults to 1. 

    Returns
    -------
    res : dict 
        A dict containing the Ensembl ID, a list of corresponding MIM IDs, and a collated MIM summary. 
    """
    mim_ids = mapping_table.query(f"ensembl_id == '{ensembl_id}'")['mim_id'].to_list()
    if len(mim_ids) == 0:
        res = {
            'ensembl_id': ensembl_id, 
            'mim_ids': None, 
            'mim_summary': None
        }
    else:
        mim_ids = [str(s).strip() for s in mim_ids]
        mim_summary = []
        for elem in mim_ids:
            mim_url = 'https://api.omim.org/api/entry?mimNumber=' 
            mim_url += elem 
            mim_url += '&include=text&include=geneMap&apiKey='
            mim_url += mim_api_key
            mim_page = requests.get(mim_url)
            time.sleep(sleep_interval)
            status_code = mim_page.status_code
            if status_code != 200:
                raise KeyError(f'Returned status code was: {status_code}')
            mim_json = xmltodict.parse(mim_page.text)
            try:
                mim_description = mim_json['omim']['entryList']['entry']['textSectionList']['textSection'][0]['textSectionContent']
            except (KeyError, TypeError):
                pass
            try:
                mim_description = mim_json['omim']['entryList']['entry']['textSectionList']['textSection']['textSectionContent']
            except (KeyError, TypeError):
                pass
            mim_description = re.sub(r' \(summary by \{[^}]*\}\)', '', mim_description)
            mim_description = re.sub(r'\{\d+:.*?\}\s*', '', mim_description)
            mim_description = add_trailing_period(mim_description)
            mim_summary.append(mim_description)
        res = {
            'ensembl_id': ensembl_id, 
            'mim_ids': mim_ids, 
            'mim_summary': mim_summary
        }
    return res
