import marimo

__generated_with = "0.23.8"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # Libraries

    Here we import all the packages we'll need to load our data and perform our analysis.
    """)
    return


@app.cell
def _():
    import os
    import shutil
    import warnings
    import anthropic
    import numpy as np
    import igraph as ig
    import pandas as pd
    import scanpy as sc
    import session_info
    import squidpy as sq
    import anndata as ad
    import matplotlib.style
    import genesetgpt as gpt
    from dotenv import load_dotenv
    import matplotlib.pyplot as plt
    from sklearn.decomposition import PCA
    from sklearn.neighbors import NearestNeighbors
    from sklearn.preprocessing import StandardScaler

    return (
        NearestNeighbors,
        PCA,
        StandardScaler,
        anthropic,
        gpt,
        ig,
        load_dotenv,
        matplotlib,
        np,
        os,
        pd,
        plt,
        sc,
        session_info,
        shutil,
        sq,
        warnings,
    )


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # Setup

    Here we set some notebook-wide options related to function verbosity, warnings, and code chunk output size.
    """)
    return


@app.cell
def _(sc, warnings):
    sc.settings.verbosity = 0
    warnings.simplefilter(action='ignore')
    pd.options.future.infer_string = False
    ad.settings.allow_write_nullable_strings = True
    mo._runtime.context.get_context().marimo_config['runtime']['output_max_bytes'] = 100_000_000
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    Next, we enable some `matplotlib` settings to make our plots look nice.
    """)
    return


@app.cell
def _(matplotlib, plt):
    matplotlib.style.use('default')
    plt.rcParams.update({
        'font.size': 12, 
        'axes.linewidth': 1.5, 
        'legend.frameon': False, 
        'figure.dpi': 320, 
        'font.family': 'Arial'
    })
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    Finally, we load our environment variables (API keys, specifically) from our `.env` dotfile.
    """)
    return


@app.cell
def _(load_dotenv):
    load_dotenv()
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # Analysis

    ## Data preprocessing

    First, we download a 10X Genomics Visium spatially-resolved transcriptomics (SRT) dataset composed of a slice of the human cortex, then make sure our `AnnData` object is set up correctly.
    """)
    return


@app.cell
def _(sq):
    ad_brain = sq.datasets.visium(sample_id='V1_Human_Brain_Section_1')
    ad_brain.layers['counts'] = ad_brain.X.copy()
    ad_brain.var_names_make_unique()
    ad_brain.var['gene'] = ad_brain.var.index.to_list()
    ad_brain.raw = ad_brain
    return (ad_brain,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    We remove the directory used to cache the downloaded `.h5ad` file so as to prevent it from clogging up our repository.
    """)
    return


@app.cell
def _(os, shutil):
    if os.path.isdir('data/V1_Human_Brain_Section_1/'):
        try: 
            shutil.rmtree('data/V1_Human_Brain_Section_1/')
        except Exception as e:
            print('Error removing the data/V1_Human_Brain_Section_1/ directory.')
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    We next perform basic spot- and gene-level QC.
    """)
    return


@app.cell
def _(ad_brain, sc):
    sc.pp.filter_cells(data=ad_brain, min_counts=1000)
    sc.pp.filter_genes(data=ad_brain, min_cells=5)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    Moving on, we select 3,000 HVGs based on the raw counts.
    """)
    return


@app.cell
def _(ad_brain, sc):
    sc.pp.highly_variable_genes(
        adata=ad_brain, 
        layer='counts', 
        n_top_genes=3000, 
        flavor='seurat_v3', 
        subset=False
    )
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    We then depth-normalize and log1p-transform the raw counts, and save the resulting matrix in a new layer in our `AnnData` object.
    """)
    return


@app.cell
def _(ad_brain, sc):
    ad_brain.X = sc.pp.normalize_total(adata=ad_brain, target_sum=1e4, inplace=False)['X']
    sc.pp.log1p(ad_brain)
    ad_brain.layers['norm'] = ad_brain.X.copy()
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    Prior to performing initial dimension reduction with PCA, we scale the normalized counts such that they have zero mean and unit variance.
    """)
    return


@app.cell
def _(ad_brain, sc):
    sc.pp.scale(ad_brain)
    sc.pp.pca(
        data=ad_brain, 
        n_comps=50, 
        random_state=312, 
        mask_var='highly_variable'
    )
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    Using the top 30 dimensions of the PCA embedding we estimate a KNN graph, then sort the graph into clusters via the Leiden algorithm.
    """)
    return


@app.cell
def _(ad_brain, sc):
    sc.pp.neighbors(
        adata=ad_brain, 
        n_neighbors=20,
        n_pcs=30,  
        use_rep='X_pca', 
        metric='cosine', 
        random_state=312
    )
    sc.tl.leiden(
        adata=ad_brain, 
        resolution=0.5, 
        flavor='igraph',
        n_iterations=2, 
        random_state=312
    )
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    We plot the Leiden clustering on our spatial coordinates:
    """)
    return


@app.cell
def _(ad_brain, plt, sq):
    sq.pl.spatial_scatter(
        adata=ad_brain, 
        shape='hex', 
        color='leiden', 
        title='Leiden', 
        img=False, 
        size=1.5
    )
    plt.gca().set_xlabel('Spatial 1')
    plt.gca().set_xlabel('Spatial 2')
    plt.show()
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    Next, we further reduce dimensionality via UMAP.
    """)
    return


@app.cell
def _(ad_brain, sc):
    sc.tl.umap(adata=ad_brain, random_state=312)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    Plotting the UMAP embedding shows distinct clusters:
    """)
    return


@app.cell
def _(ad_brain, plt, sc):
    sc.pl.embedding(
        adata=ad_brain, 
        basis='umap', 
        color='leiden', 
        title='Leiden',
        frameon=True, 
        size=30, 
        alpha=0.8, 
        show=False
    )
    plt.gca().set_xlabel('UMAP 1')
    plt.gca().set_ylabel('UMAP 2')
    plt.gca().spines[['right', 'top']].set_visible(False)
    plt.show()
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    Next, we estimate a set of spatial neighbors for each spot.
    """)
    return


@app.cell
def _(ad_brain, sq):
    sq.gr.spatial_neighbors(adata=ad_brain, n_neighs=6)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    We extract a `list` of the top 3,000 HVGs, then test them for spatial structure using a Moran's I test.
    """)
    return


@app.cell
def _(ad_brain, sq):
    top3k_hvgs = ad_brain.var[ad_brain.var['highly_variable']]['gene'].to_list()
    sq.gr.spatial_autocorr(
        adata=ad_brain,
        mode='moran',
        genes=top3k_hvgs, 
        use_raw=False, 
        layer='norm', 
        n_perms=100,
        n_jobs=4, 
        seed=312
    )
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    After extracting the table of test results, we remove genes that exhibit no statistically significant spatial dependence, classify the top 1,000 remaining genes as SVGs, and add a flag for spatial variability to our `AnnData` object.
    """)
    return


@app.cell
def _(ad_brain):
    moran_df = ad_brain.uns['moranI'].copy()
    moran_df.query('pval_sim_fdr_bh < 0.05', inplace=True)
    moran_df.sort_values(
        by='I',
        key=lambda col: col.abs(),
        ascending=False, 
        inplace=True
    )
    top1k_svgs = moran_df.index.to_list()[:1000]
    ad_brain.var['spatially_variable'] = ad_brain.var_names.isin(values=top1k_svgs)
    return (top1k_svgs,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    Moving on, we extract a matrix of normalized counts with genes as rows and spots as columns, then scale it.
    """)
    return


@app.cell
def _(StandardScaler, ad_brain, top1k_svgs):
    expr_mtx = ad_brain[:, top1k_svgs].layers['norm'].T.toarray()
    scaler = StandardScaler(with_mean=True, with_std=True)
    expr_mtx_scaled = scaler.fit_transform(X=expr_mtx)
    return (expr_mtx_scaled,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    We then reduce dimensionality of the scaled SVG expression matrix to 30 dimensions via PCA.
    """)
    return


@app.cell
def _(PCA, expr_mtx_scaled):
    pca = PCA(n_components=30, random_state=312)
    pc_mtx = pca.fit_transform(X=expr_mtx_scaled)
    return (pc_mtx,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    We estimate a KNN graph in PCA space, convert it to an adjacency matrix, and utilize the Leiden algorithm to sort the graph into clusters of SVGs with similar patterns. Lastly, we create a `DataFrame` with the clustering results.
    """)
    return


@app.cell
def _(NearestNeighbors, ig, np, pc_mtx, pd, top1k_svgs):
    nns = NearestNeighbors(n_neighbors=20, metric='cosine').fit(X=pc_mtx)
    knn_graph = nns.kneighbors_graph(X=pc_mtx, mode='connectivity')
    adj_mtx = knn_graph.toarray()
    adj_mtx = np.maximum(adj_mtx, adj_mtx.T)
    g = ig.Graph.Adjacency((adj_mtx > 0).tolist(), mode=ig.ADJ_UNDIRECTED)
    partition = g.community_leiden(resolution=0.01)
    cluster_df = pd.DataFrame(data={
        'gene': top1k_svgs, 
        'leiden': np.array(partition.membership)
    })
    return (cluster_df,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    Let's take a look at how many SVGs belong to each cluster (or module):
    """)
    return


@app.cell
def _(cluster_df):
    cluster_df['leiden'].value_counts(sort=False)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    Next, we create a `dict` containing the assigned SVGs for each module.
    """)
    return


@app.cell
def _(cluster_df):
    module_gene_dict = {
        cl: cluster_df.query(f'leiden == {cl}')['gene'].to_list()
        for cl in cluster_df['leiden'].unique()
    }
    return (module_gene_dict,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    We then score each module using the normalized counts, and add the per-spot scores to our `AnnData` object.
    """)
    return


@app.cell
def _(ad_brain, module_gene_dict, sc):
    for cl, genes in module_gene_dict.items():
        sc.tl.score_genes(
            adata=ad_brain,
            gene_list=genes,
            score_name=f'svg_module_{cl}',
            random_state=312,
            use_raw=False,
            layer='norm'
        )
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    We plot the resulting module scores on the spatial coordinates below:
    """)
    return


@app.cell
def _(ad_brain, cluster_df, plt, sq):
    sq.pl.spatial_scatter(
        adata=ad_brain,
        shape='hex',
        size=1.5, 
        color=[f'svg_module_{c}' for c in list(set(cluster_df['leiden']))],
        img=False
    )
    plt.show()
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## AI module summarization

    Before anything else, we need to load our API keys that we'll use to access various databases and LLMs.
    """)
    return


@app.cell
def _(os):
    mim_key = os.getenv('MIM_API_KEY')
    entrez_key = os.getenv('ENTREZ_API_KEY')
    claude_key = os.getenv('ANTHROPIC_API_KEY')
    return claude_key, entrez_key, mim_key


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    Now we're ready to move on to the interesting bit - summarizing our SVG modules functionally using LLMs. We start by fetching some gene-level data we'll need to perform the analysis.
    """)
    return


@app.cell
def _(gpt):
    all_hs_genes = gpt.fetch_gene_table()
    mim_table = gpt.fetch_mim_table()
    return all_hs_genes, mim_table


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    We build a per-gene prompt based on information dynamically pulled from databases such as the Human Protein Atlas, Entrez, etc. that will be later used to coalesce those summaries into a concise gene-level summary.
    """)
    return


@app.cell
def _(all_hs_genes, entrez_key, gpt, mim_key, mim_table, top1k_svgs):
    user_prompt_df = gpt.build_prompt_df(
        gene_list=top1k_svgs, 
        gene_id_table=all_hs_genes, 
        mim_mapping_table=mim_table, 
        mim_api_key=mim_key, 
        entrez_email='j.leary@ufl.edu', 
        entrez_api_key=entrez_key
    )
    return (user_prompt_df,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    Next, we define our system prompt, which helps determine the overall style and form of the LLM results while providing important biological context.
    """)
    return


@app.cell
def _():
    prompt_system = 'You are an experienced computational biologist with extensive knowledge of transcriptomics analyses such as single-cell RNA-seq and spatially-resolved transcriptomics. When generating responses, you consider the statistical, computational, and biological angles of the question at hand. Your responses are detailed without being too overly technical. The system being studied is the healthy human cortex, and the data were assayed using 10X Genomics Visium V1. Quality control, pre-processing, and analysis were performed with a workflow based on the Scanpy and Squidpy Python packages.'
    return (prompt_system,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    Using our environment variables we loaded at the beginning, we generate an Anthropic (the Claude LLM provider) client with our API key.
    """)
    return


@app.cell
def _(anthropic, claude_key):
    claude_client = anthropic.Anthropic(api_key=claude_key)
    return (claude_client,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    Once again using parallel procesing to speed things up, we ask Claude Haiku 4.5 to describe each gene's functionality in our dataset based on the functional summary prompts we just built.
    """)
    return


@app.cell
def _(claude_client, gpt, prompt_system, user_prompt_df):
    gene_sumys = gpt.summarize_individual_genes(
        user_prompt_df=user_prompt_df, 
        provider='anthropic', 
        client=claude_client,
        model='claude-haiku-4-5', 
        prompt_system=prompt_system, 
        n_max_tokens=750, 
        n_workers=4
    )
    return (gene_sumys,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    For the final summarization step, we write a loop that generates the paragraph-length summary, name, and confidence score for every SVG module. We also save a string of the raw JSON generated by the LLM when summarizing each module.
    """)
    return


@app.cell
def _(claude_client, cluster_df, gene_sumys, gpt, pd, prompt_system):
    unique_svg_modules = list(set(cluster_df['leiden']))
    module_summaries = []
    module_jsons = []
    for module in unique_svg_modules:
        module_genes = cluster_df.query(expr='leiden == @module')['gene'].to_list()
        module_sumy = gpt.summarize_module(
            module_genes=module_genes, 
            gene_sumy_df=gene_sumys, 
            prompt_system=prompt_system, 
            provider='anthropic', 
            client=claude_client, 
            model='claude-haiku-4-5',
            n_max_tokens=3000
        )
        module_sumy_df = module_sumy['module_summary_df']
        module_sumy_df['module_id'] = module
        module_summaries.append(module_sumy_df)
        module_jsons.append(module_sumy['model_json'])
    final_summary_df = pd.concat(module_summaries, ignore_index=True)
    return (final_summary_df,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    Finally, let's check out the results for the first module:
    """)
    return


@app.cell
def _(final_summary_df):
    mo.md(final_summary_df['summary'].to_list()[0])
    return


@app.cell
def _(final_summary_df):
    mo.md(final_summary_df['name'].to_list()[0])
    return


@app.cell
def _(final_summary_df):
    print(f"Confidence score: {final_summary_df['score'].to_list()[0]}")
    return


@app.cell
def _(final_summary_df):
    mo.md(final_summary_df['score_rationale'].to_list()[0])
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # Session information
    """)
    return


@app.cell
def _(session_info):
    session_info.show(cpu=True)
    return


if __name__ == "__main__":
    app.run()
