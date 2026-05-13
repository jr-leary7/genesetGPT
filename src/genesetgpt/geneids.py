import numpy as np
import pandas as pd
from typing import Optional
from pybiomart import Server

def fetch_gene_table(sort_by: str = 'hgnc_symbol', alt_ensembl_archive: Optional[str] = None) -> pd.DataFrame: 
    """
    Fetch a table of gene IDs and other per-gene metadata. 

    Parameters
    ----------
    sort_by : str 
        A string specifying the gene ID to sort the table by. Defaults to 'hgnc_symbol'. 
    alt_ensembl_archive : str 
        An optional string specifying a date-specific alternative Ensembl archive URL to query e.g., 'http://may2025.archive.ensembl.org' (the most recent working archive as of this implementation). Try using this argument if the default Ensembl server times out or responds with an error. Defaults to None. 

    Returns
    -------
    gene_df : pd.DataFrame 
        A DataFrame containing gene IDs, biotypes, descriptions, and chromosomal locations. 
    """
    if alt_ensembl_archive is not None:
        server = Server(host=alt_ensembl_archive)
    else:
        server = Server(host='http://www.ensembl.org')
    mart = server.marts['ENSEMBL_MART_ENSEMBL']
    dataset = mart.datasets['hsapiens_gene_ensembl']
    gene_df = dataset.query(
        attributes=[
            'ensembl_gene_id', 
            'entrezgene_id', 
            'hgnc_symbol', 
            'description', 
            'gene_biotype', 
            'chromosome_name', 
            'start_position', 
            'end_position'
        ],
        only_unique=True
    )
    gene_df.rename(
        columns={
            'Gene stable ID': 'ensembl_id', 
            'HGNC symbol': 'hgnc_symbol', 
            'NCBI gene (formerly Entrezgene) ID': 'entrez_id', 
            'Gene description': 'description', 
            'Gene type': 'biotype', 
            'Chromosome/scaffold name': 'chromosome', 
            'Gene start (bp)': 'location_start', 
            'Gene end (bp)': 'location_end'
        }, 
        inplace=True
    )
    gene_df['entrez_id'] = gene_df['entrez_id'].map(lambda x: f'{x:.0f}').astype(str)
    gene_df['description'] = gene_df['description'].str.replace(
        pattern=r'\s*\[Source:[^\]]*\]', 
        repl='', 
        regex=True
    )
    gene_df.replace(
        to_replace=['nan', 'None'],
        value=np.nan,
        inplace=True
    )
    if sort_by not in gene_df.columns: 
        raise KeyError(f"`sort_by='{sort_by}'` is not a valid column.")
    gene_df.sort_values(by=sort_by, inplace=True)
    return gene_df 
