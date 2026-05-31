import importlib.resources as pkg_resources

def load_example_gene_set():
    """
    Loads a list of the top 50 B-cell marker genes generated from the widely-used 10X Genomics PBM3k scRNA-seq dataset.
    
    Returns
    -------
        A list of HGNC symbols representing the 50 most-significant marker genes for the B-cell cluster. 
    """
    data_path = pkg_resources.files(anchor='genesetgpt.data').joinpath('example_gene_set.txt')
    with data_path.open(mode='r', encoding='utf-8') as file:
        genes = [line.strip() for line in file if line.strip()]
    return genes
