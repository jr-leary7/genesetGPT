import importlib.resources as pkg_resources

def load_example_gene_set():
    """
    Load a list of B-cell marker genes generated from the widely-used `10X Genomics PBMC3k scRNA-seq dataset`_.
    
    Returns
    -------
        A list of HGNC symbols representing the 50 most-significant marker genes by Wilcoxon test for the B-cell cluster. 

    .. _10X Genomics PBMC3k scRNA-seq dataset: https://scanpy.scverse.org/en/stable/tutorials/basics/clustering-2017.html
    """
    data_path = pkg_resources.files(anchor='genesetgpt.data').joinpath('example_gene_set.txt')
    with data_path.open(mode='r', encoding='utf-8') as file:
        genes = [line.strip() for line in file if line.strip()]
    return genes
