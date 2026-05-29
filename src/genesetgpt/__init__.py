__version__ = '0.1.2'

from .hpa import fetch_HPA_data
from .geneids import fetch_gene_table
from .entrez import fetch_entrez_summary
from .datasets import load_example_gene_set
from .mim import fetch_mim_table, fetch_mim_summary 
from .prompt import build_user_prompt, build_prompt_df
from .utils import add_trailing_period, cosine_sim, get_aliases
from .llm import summarize_gene, get_embedding, summarize_individual_genes, summarize_module
from .uniprot import fetch_canonical_protein_product, clean_uniprot_summary, fetch_uniprot_summary

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
    'build_prompt_df', 
    'summarize_gene', 
    'summarize_individual_genes',
    'summarize_module', 
    'get_embedding', 
    'add_trailing_period', 
    'cosine_sim', 
    'get_aliases', 
    'load_example_gene_set'
]
