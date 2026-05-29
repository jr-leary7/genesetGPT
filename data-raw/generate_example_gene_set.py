##### libraries #####
import os
import pandas as pd
import scanpy as sc

##### basic pre-processing & qc filtering #####
ad_pbmc = sc.datasets.pbmc3k()
ad_pbmc.var_names_make_unique()
ad_pbmc.obs['cell'] = ad_pbmc.obs.index.to_list()
ad_pbmc.var['gene'] = ad_pbmc.var.index.to_list()
ad_pbmc.raw = ad_pbmc
sc.pp.filter_cells(data=ad_pbmc, min_genes=200)
sc.pp.filter_genes(data=ad_pbmc, min_cells=3)
ad_pbmc.var['mt'] = ad_pbmc.var_names.str.startswith(pat='MT-')
sc.pp.calculate_qc_metrics(
    adata=ad_pbmc, 
    qc_vars=['mt'], 
    percent_top=None, 
    log1p=False, 
    inplace=True
)
ad_pbmc = ad_pbmc[
    (ad_pbmc.obs.n_genes_by_counts < 2500) & (ad_pbmc.obs.n_genes_by_counts > 200) & (ad_pbmc.obs.pct_counts_mt < 5), :,
].copy()
ad_pbmc.layers['counts'] = ad_pbmc.X.copy()
sc.pp.normalize_total(adata=ad_pbmc, target_sum=1e4)
sc.pp.log1p(ad_pbmc)

##### hvg identification, scaling, & pca #####
sc.pp.highly_variable_genes(
    adata=ad_pbmc,
    layer='counts',
    n_top_genes=3000,
    flavor='seurat_v3'
)
ad_pbmc.layers['scaled'] = ad_pbmc.X.toarray()
sc.pp.regress_out(
    adata=ad_pbmc, 
    keys=['total_counts', 'pct_counts_mt'], 
    layer='scaled'
)
sc.pp.scale(
    ad_pbmc, 
    max_value=10, 
    layer='scaled'
)
sc.pp.pca(
    data=ad_pbmc, 
    layer='scaled', 
    n_comps=50, 
    mask_var='highly_variable', 
    random_state=312
)

##### leiden clustering #####
sc.pp.neighbors(
    adata=ad_pbmc,
    n_neighbors=20, 
    use_rep='X_pca', 
    n_pcs=30, 
    metric='cosine', 
    random_state=312
)
sc.tl.leiden(
    adata=ad_pbmc,
    resolution=0.7,
    random_state=312,
    flavor='igraph',
    n_iterations=2,
    directed=False
)

##### de gene identification #####
sc.tl.rank_genes_groups(
    adata=ad_pbmc, 
    groupby='leiden', 
    mask_var='highly_variable', 
    method='wilcoxon', 
    use_raw=False, 
    random_state=312, 
    n_jobs=2
) 
de_genes = sc.get.rank_genes_groups_df(adata=ad_pbmc, group='2').head(50)['names'].to_list()

##### save example gene dataset #####
target_path = os.path.join('..', 'src', 'genesetgpt', 'data', 'example_gene_set.txt')
os.makedirs(os.path.dirname(target_path), exist_ok=True)
with open(target_path, 'w', encoding='utf-8') as file:
    for gene in de_genes:
        file.write(f'{gene}\n')
