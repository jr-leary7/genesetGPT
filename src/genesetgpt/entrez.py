import re 
from Bio import Entrez
from typing import TypedDict

class EntrezSummary(TypedDict):
    entrez_id: str
    entrez_summary: str

def fetch_entrez_summary(entrez_id: str, 
                         entrez_email: str, 
                         entrez_api_key : str = None) -> EntrezSummary:
    """
    Fetch the NCBI Entrez summary for a given gene. 

    Parameters
    ----------
    entrez_id : ``str ``
        A string specifying the Entrez ID of the gene of interest. 
    entrez_email : ``str``
        A string specifying your email address that will be associated with the Entrez query. 
    entrez_api_key : ``str``
        A string specifying your `optional API key`_ for the Entrez database. Providing your key allows more API requests per second (10 versus 3). Defaults to None.

    Returns
    -------
        A dictionary containing the gene's Entrez ID and corresponding summary.

    .. _optional API key: https://support.nlm.nih.gov/kbArticle/?pn=KA-05317
    """
    Entrez.email = entrez_email
    if entrez_api_key is not None:
        Entrez.api_key = entrez_api_key
    entrez_xml = Entrez.esummary(
        db='gene', 
        id=entrez_id, 
        retmode='xml'
    )
    entrez_records = Entrez.read(source=entrez_xml, validate=False)
    entrez_xml.close()
    try:
        entrez_summary = entrez_records['DocumentSummarySet']['DocumentSummary'][0]['Summary']
        entrez_summary = re.sub(
            pattern=r'\[provided by RefSeq[^\]]*\]', 
            repl='', 
            string=entrez_summary
        )
    except Exception as e:
        entrez_summary = None
    res = {
        'entrez_id': entrez_id, 
        'entrez_summary': entrez_summary
    }
    return res
