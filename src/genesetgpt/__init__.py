__version__ = '0.1.0'

from .geneids import fetch_gene_table
from .hpa import fetch_HPA_data
from .uniprot import fetch_canonical_protein_product, clean_uniprot_summary, fetch_uniprot_summary
from .entrez import fetch_entrez_summary
from .mim import fetch_mim_table, fetch_mim_summary 
from .prompt import build_user_prompt
from .llm import summarize_genes, get_embedding 
from .utils import add_trailing_period, cosine_sim

__all__ = [
    'fetch_gene_table', 
    'fetch_HPA_data', 
    'fetch_canonical_protein_product', 
    'clean_uniprot_summary', 
    'fetch_uniprot_summary', 
    'fetch_entrez_summary', 
    'fetch_mim_table', 
    'fetch_mim_summary', 
    'build_user_prompt', 
    'summarize_genes', 
    'get_embedding', 
    'add_trailing_period', 
    'cosine_sim'
]