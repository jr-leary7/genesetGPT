import re 
from Bio import Entrez
from typing import TypedDict

class EntrezSummary(TypedDict):
    entrez_id: str
    entrez_summary: str

def fetch_entrez_summary(entrez_id: str, entrez_email: str) -> EntrezSummary:
    """
    Fetch the NCBI Entrez summary for a given gene. 

    Parameters
    ----------
    entrez_id : str 
        A string specifying the Entrez ID of the gene of interest. 
    entrez_email : str 
        A string specifying the email address associated with the Entrez query. 
    """
    Entrez.email = entrez_email
    entrez_xml = Entrez.esummary(
        db='gene', 
        id=entrez_id, 
        retmode='xml'
    )
    entrez_records = Entrez.read(entrez_xml, validate=False)
    entrez_xml.close()
    try:
        entrez_summary = entrez_records['DocumentSummarySet']['DocumentSummary'][0]['Summary']
        entrez_summary = re.sub(r'\[provided by RefSeq[^\]]*\]', '', entrez_summary)
    except Exception as e:
        entrezy_summary = None
    res = {
        'entrez_id': entrez_id, 
        'entrez_summary': entrez_summary
    }
    return res