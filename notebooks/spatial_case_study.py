import marimo

__generated_with = "0.18.1"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Libraries

    Here we import all the packages we'll need to load our data and perform our analysis.
    """)
    return


@app.cell
def _():
    import os
    import re
    import time
    import json
    import openai
    import shutil
    import pickle
    import getpass
    import warnings
    import numpy as np
    import igraph as ig
    import scanpy as sc
    import pandas as pd
    import marimo as mo
    import session_info
    import squidpy as sq
    import anndata as ad
    import genesetgpt as gpt
    from openai import OpenAI
    from datetime import datetime
    from functools import partial
    from pydantic import BaseModel
    from dotenv import load_dotenv
    import matplotlib.pyplot as plt
    from pandarallel import pandarallel
    from sklearn.decomposition import PCA
    from sklearn.neighbors import NearestNeighbors
    from sklearn.preprocessing import StandardScaler
    from concurrent.futures import ThreadPoolExecutor
    return (
        BaseModel,
        NearestNeighbors,
        OpenAI,
        PCA,
        StandardScaler,
        ThreadPoolExecutor,
        datetime,
        getpass,
        gpt,
        ig,
        json,
        load_dotenv,
        mo,
        np,
        os,
        pandarallel,
        partial,
        pd,
        plt,
        sc,
        session_info,
        shutil,
        sq,
        warnings,
    )


@app.cell
def _(sc, warnings):
    sc.settings.verbosity = 0
    warnings.simplefilter('ignore', category=UserWarning)
    warnings.simplefilter('ignore', category=FutureWarning)
    return


@app.cell
def _(load_dotenv):
    load_dotenv(dotenv_path='../.env')
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Data

    Here we load in a 10X Genomics Visium dataset containing a slice of the human cerebral cortex ([source](https://cf.10xgenomics.com/samples/spatial-exp/1.1.0/V1_Human_Brain_Section_1/V1_Human_Brain_Section_1_web_summary.html)).
    """)
    return


@app.cell
def _(sq):
    ad_brain = sq.datasets.visium(sample_id='V1_Human_Brain_Section_1')
    ad_brain.layers['counts'] = ad_brain.X.copy()
    ad_brain.var['gene'] = ad_brain.var.index.to_list()
    ad_brain.var_names_make_unique()
    ad_brain.raw = ad_brain
    return (ad_brain,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Since downloading the dataset using `squidpy` creates a cache directory called `data/` in our current directory, we remove it (if the directory exists). This is done because we don't want to accidentally commit a large data file to our GitHub repository.
    """)
    return


@app.cell
def _(os, shutil):
    if os.path.isdir('data/'):
        try: 
            shutil.rmtree('data/')
        except Exception as e:
            print('Error removing the data/ directory.')
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Analysis

    ## Preprocessing the spatial data

    We start by performing some basic spot- and gene-level QC.
    """)
    return


@app.cell
def _(ad_brain, sc):
    sc.pp.filter_cells(ad_brain, min_counts=1000)
    sc.pp.filter_genes(ad_brain, min_cells=5)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    We continue by identifying a set of 3,000 naive HVGs.
    """)
    return


@app.cell
def _(ad_brain, sc):
    sc.pp.highly_variable_genes(
        ad_brain, 
        n_top_genes=3000, 
        flavor='seurat_v3', 
        subset=False
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Next, we depth-normalize and log1p-transform the raw counts.
    """)
    return


@app.cell
def _(ad_brain, sc):
    ad_brain.X = sc.pp.normalize_total(ad_brain, target_sum=1e4, inplace=False)['X']
    sc.pp.log1p(ad_brain)
    ad_brain.layers['norm'] = ad_brain.X.copy()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Moving on, we scaled the normalized counts and run PCA using our HVG set.
    """)
    return


@app.cell
def _(ad_brain, sc):
    sc.pp.scale(ad_brain)
    sc.tl.pca(
        ad_brain, 
        n_comps=50, 
        random_state=312, 
        mask_var='highly_variable'
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    In PCA space we identify a shared nearest-neighbors (SNN) graph with $k = 20$ neighbors per spot. We then partition the graph into clusters using the Leiden algorithm.
    """)
    return


@app.cell
def _(ad_brain, sc):
    sc.pp.neighbors(
        ad_brain, 
        n_neighbors=20,
        n_pcs=30,  
        use_rep='X_pca', 
        metric='cosine', 
        random_state=312
    )
    sc.tl.leiden(
        ad_brain, 
        resolution=0.5, 
        flavor='igraph',
        n_iterations=2, 
        random_state=312
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Here we plot the first two dimensions of the PCA embedding, with spots colored by Leiden cluster ID.
    """)
    return


@app.cell
def _(ad_brain, plt, sc):
    sc.pl.embedding(
        ad_brain, 
        basis='pca', 
        color='leiden',
        title='Leiden',
        frameon=True, 
        size=30, 
        alpha=0.75,
        show=False
    )
    plt.gca().set_xlabel('PC 1')
    plt.gca().set_ylabel('PC 2')
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Now we run UMAP to further reduce dimensionality down to 2D for visualization purposes.
    """)
    return


@app.cell
def _(ad_brain, sc):
    sc.tl.umap(ad_brain, random_state=312)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    We plot the clustering on the UMAP embedding\:
    """)
    return


@app.cell
def _(ad_brain, plt, sc):
    sc.pl.embedding(
        ad_brain, 
        basis='umap', 
        color='leiden',
        title='Leiden',
        frameon=True, 
        size=30, 
        alpha=0.75,
        show=False
    )
    plt.gca().set_xlabel('UMAP 1')
    plt.gca().set_ylabel('UMAP 2')
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    When plotting the clustering on the spatial coordinates, we see clear layers\:
    """)
    return


@app.cell
def _(ad_brain, plt, sq):
    sq.pl.spatial_scatter(
        ad_brain, 
        shape='hex',
        color='leiden', 
        title='Leiden'
    )
    plt.gca().set_xlabel('Spatial 1')
    plt.gca().set_ylabel('Spatial 2')
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Now we identify a set of spatial neighbors for every spot with $k = 10$.
    """)
    return


@app.cell
def _(ad_brain, sq):
    sq.gr.spatial_neighbors(ad_brain, n_neighs=10)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Using the naive HVG set of 3,000 genes, we test for spatial autocorrelation via [Moran's I test](https://en.wikipedia.org/wiki/Moran%27s_I).
    """)
    return


@app.cell
def _(ad_brain, sq):
    top3k_hvgs = ad_brain.var[ad_brain.var['highly_variable']]['gene'].to_list()
    sq.gr.spatial_autocorr(
        ad_brain,
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
def _(mo):
    mo.md(r"""
    After filtering out genes with adjusted *p*-values greater than 0.05, we identify the top 500 most spatially variable genes. We also add a Boolean flag to our `AnnData` object specifying which genes are classified as SVGs.
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
    top500_svgs = moran_df.index.to_list()[:500]
    ad_brain.var['spatially_variable'] = ad_brain.var_names.isin(top500_svgs)
    moran_df
    return (top500_svgs,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Next, we extract the normalized counts matrix for the SVGs and scale it.
    """)
    return


@app.cell
def _(StandardScaler, ad_brain, top500_svgs):
    expr_mtx = ad_brain[:, top500_svgs].layers['norm'].T.toarray()
    scaler = StandardScaler(with_mean=True, with_std=True)
    expr_mtx_scaled = scaler.fit_transform(expr_mtx)
    return (expr_mtx_scaled,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    With the scaled SVG matrix in hand, we run PCA and reduce dimensionality down to 30 components.
    """)
    return


@app.cell
def _(PCA, expr_mtx_scaled):
    pca = PCA(n_components=30, random_state=312)
    pc_mtx = pca.fit_transform(expr_mtx_scaled)
    return (pc_mtx,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The next step is to create a *k* nearest-neighbors (KNN) graph with $k = 20$ neighbors per gene, after which we convert it to an undirected adjacency matrix, which we then partition into clusters using the Leiden algorithm.
    """)
    return


@app.cell
def _(NearestNeighbors, ig, np, pc_mtx):
    nns = NearestNeighbors(n_neighbors=20, metric='cosine').fit(pc_mtx)
    knn_graph = nns.kneighbors_graph(pc_mtx, mode='connectivity')
    adj_mtx = knn_graph.toarray()
    adj_mtx = np.maximum(adj_mtx, adj_mtx.T)
    g = ig.Graph.Adjacency((adj_mtx > 0).tolist(), mode=ig.ADJ_UNDIRECTED)
    partition = g.community_leiden(resolution=0.02)
    return (partition,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Finally, we create a `DataFrame` of the gene-to-cluster assignments.
    """)
    return


@app.cell
def _(np, partition, pd, top500_svgs):
    cluster_df = pd.DataFrame({
        'gene': top500_svgs, 
        'leiden': np.array(partition.membership)
    })
    return (cluster_df,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    We extract lists of the genes assigned to each cluster, which we'll use to perform gene set scoring.
    """)
    return


@app.cell
def _(cluster_df):
    genes_clust0 = cluster_df.query('leiden == 0')['gene'].to_list()
    genes_clust1 = cluster_df.query('leiden == 1')['gene'].to_list()
    genes_clust2 = cluster_df.query('leiden == 2')['gene'].to_list()
    return genes_clust0, genes_clust1, genes_clust2


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    For each SVG cluster we perform gene set scoring using [`sc.tl.score_genes()`](https://scanpy.readthedocs.io/en/stable/generated/scanpy.tl.score_genes.html), which adds a column to `.obs` with a measurement of how highly-expressed the gene set is compared to a randomly sampled background.
    """)
    return


@app.cell
def _(ad_brain, genes_clust0, genes_clust1, genes_clust2, sc):
    sc.tl.score_genes(
        ad_brain, 
        gene_list=genes_clust0,
        score_name='svg_cluster0', 
        random_state=312, 
        use_raw=False
    )
    sc.tl.score_genes(
        ad_brain, 
        gene_list=genes_clust1,
        score_name='svg_cluster1', 
        random_state=312, 
        use_raw=False
    )
    sc.tl.score_genes(
        ad_brain, 
        gene_list=genes_clust2,
        score_name='svg_cluster2', 
        random_state=312, 
        use_raw=False
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    We plot the gene set scores for each SVG cluster below\:
    """)
    return


@app.cell
def _(ad_brain, plt, sq):
    svg_clusters = [0, 1, 2]
    plot_titles = [f'SVG Cluster {c}' for c in svg_clusters]
    fig, axes = plt.subplots(
        nrows=1, 
        ncols=3, 
        figsize = (15, 5), 
        sharex=True, 
        sharey=True
    )
    for ax, c, title in zip(axes, svg_clusters, plot_titles):
        sq.pl.spatial_scatter(
            ad_brain,
            shape='hex',
            color=f'svg_cluster{c}',
            title=title,
            ax=ax
        )
        ax.set_xlabel('Spatial 1')
        ax.set_ylabel('Spatial 2')
    fig.tight_layout()
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## LLM Summarization

    To summarize our gene sets using LLMs we'll need some other data sources. We start by using our `geneSetGPT` package to retrieve a table containing various IDs for all the genes in the human genome.
    """)
    return


@app.cell
def _(gpt):
    all_hs_genes = gpt.fetch_gene_table()
    all_hs_genes
    return (all_hs_genes,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Next, we use our package to fetch a table containing a mapping from Mendelian Inheritance of Man (MIM) IDs to other gene IDs.
    """)
    return


@app.cell
def _(gpt):
    mim_table = gpt.fetch_mim_table()
    mim_table
    return (mim_table,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    We filter our human gene ID table to include just the SVG set\:
    """)
    return


@app.cell
def _(all_hs_genes):
    svg_gene_ids = all_hs_genes.query('hgnc_symbol in @top500_svgs').copy()
    svg_gene_ids.dropna(inplace=True)
    svg_gene_ids
    return (svg_gene_ids,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Now we put it all together and add the user prompt for each gene to the `DataFrame` containing our SVG gene information, complete with parallelism and a progress bar - both powered by the `pandarallel` package. **Note**: this will take a while since we're iterating over 500+ genes (hence the parallel processing and progress bar).
    """)
    return


@app.cell
def _(pandarallel):
    pandarallel.initialize(
        progress_bar=True, 
        nb_workers=3, 
        verbose=0
    )
    return


@app.cell
def _(gpt, mim_table, os, svg_gene_ids):
    svg_gene_ids['prompt_user'] = svg_gene_ids.parallel_apply(
        lambda row: 
        gpt.build_user_prompt(
            ensembl_id=row['ensembl_id'], 
            hgnc_symbol=row['hgnc_symbol'], 
            entrez_id=row['entrez_id'], 
            entrez_email='j.leary@ufl.edu', 
            mim_mapping_table=mim_table, 
            mim_api_key=os.getenv('MIM_API_KEY'), 
            include_aliases=True
        ), 
        axis=1
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Let's take a look at the dataset annotated with user prompts\:
    """)
    return


@app.cell
def _(svg_gene_ids):
    svg_gene_ids
    return


@app.cell
def _(mo, svg_gene_ids):
    mo.md(svg_gene_ids['prompt_user'].to_list()[0])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    We'll also need to write out the development prompt, which tells the model what kind of tone and style to adopt when generating answers.
    """)
    return


@app.cell
def _():
    prompt_dev = 'You are an experienced computational biologist with advanced knowledge of transcriptomics analyses such as single-cell RNA-seq and spatial transcriptomics. When generating responses, you consider the statistical, computational, and biological angles of the question at hand. Your responses are detailed without being too overly technical.'
    return (prompt_dev,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    We set up our OpenAI client using our API key below\:
    """)
    return


@app.cell
def _(OpenAI, os):
    client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
    return (client,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    We execute the prompts in parallel with 4 workers, since the process is I/O bound via HTTP requests.
    """)
    return


@app.cell
def _(ThreadPoolExecutor, client, gpt, partial, prompt_dev, svg_gene_ids):
    summarize_one = partial(
        gpt.summarize_genes,
        prompt_dev=prompt_dev,
        openai_client=client,
        openai_model='gpt-4o-mini'
    )
    user_prompts = svg_gene_ids['prompt_user'].to_list()
    with ThreadPoolExecutor(max_workers=4) as pool:
        results = list(pool.map(summarize_one, user_prompts))
    llm_summaries, llm_scores = zip(*results)
    return llm_scores, llm_summaries


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    We add the LLM summaries and confidence scores to our `DataFrame`.
    """)
    return


@app.cell
def _(llm_scores, llm_summaries, svg_gene_ids):
    svg_gene_ids['llm_summary'] = llm_summaries
    svg_gene_ids['llm_confidence_score'] = llm_scores
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    We define a short class for our gene set-level summaries.
    """)
    return


@app.cell
def _(BaseModel):
    class GeneSetSummary(BaseModel):
        summary: str
        name: str
        confidence: float
    return (GeneSetSummary,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Here we set up a `list` object containing our unique cluster IDs.
    """)
    return


@app.cell
def _(cluster_df):
    unique_svg_clusters = list(set(cluster_df['leiden']))
    return (unique_svg_clusters,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Now we can iterate over our cluster IDs, set up the queries to the LLM, and generate responses containing 1) a summary of the gene set 2) a name for the gene set and 3) an estimated confidence score.

    We also repeat the summarization, naming, and scoring process *without* the additional functional context in order to perform a comparison.
    """)
    return


@app.cell
def _(
    GeneSetSummary,
    client,
    cluster_df,
    prompt_dev,
    svg_gene_ids,
    unique_svg_clusters,
):
    cluster_summaries = []
    cluster_summaries_no_context = []
    cluster_names = []
    cluster_names_no_context = []
    cluster_scores = []
    cluster_scores_no_context = []
    model_jsons = []
    model_jsons_no_context = []
    for clust in unique_svg_clusters:
        cluster_genes = cluster_df.query(f'leiden == {clust}')['gene'].to_list()
        cluster_genes_str = ', '.join(cluster_genes)
        cluster_gene_ids = svg_gene_ids.query('hgnc_symbol in @cluster_genes').copy()
        cluster_user_prompts = cluster_gene_ids['prompt_user'].to_list()
        cluster_llm_summaries_bulleted = '\n'.join(f'- {s}' for s in cluster_user_prompts)
        summary_prompt = f"""
        Below are brief, independent descriptions of genes in a set:

        {cluster_llm_summaries_bulleted}

        Please write a concise (5–7 sentence) paragraph summarizing the common function(s) of this gene set. In addition, please provide a 0-1 score estimating how confident you are in your overall annotation. Lastly, provide a short 2-5 word name for the gene set based on your annotation.
        """
        summary_response = client.responses.parse(
            model='gpt-4.1-mini', 
            input=[
                {'role': 'developer', 'content': prompt_dev}, 
                {'role': 'user', 'content': summary_prompt}
            ], 
            text_format=GeneSetSummary
        )
        cluster_summaries.append(summary_response.output_parsed.summary)
        cluster_names.append(summary_response.output_parsed.name)
        cluster_scores.append(summary_response.output_parsed.confidence)
        model_jsons.append(summary_response.model_dump_json())
        summary_prompt_no_context = f"""
        Here is a set of genes:

        {cluster_genes_str}.

        Please write a concise (5–7 sentence) paragraph summarizing the common function(s) of this gene set. In addition, please provide a 0-1 score estimating how confident you are in your overall annotation. Lastly, provide a short 2-5 word name for the gene set based on your annotation.
        """
        summary_response_no_context = client.responses.parse(
            model='gpt-4.1-mini', 
            input=[
                {'role': 'developer', 'content': prompt_dev}, 
                {'role': 'user', 'content': summary_prompt_no_context}
            ], 
            text_format=GeneSetSummary
        )
        cluster_summaries_no_context.append(summary_response_no_context.output_parsed.summary)
        cluster_names_no_context.append(summary_response_no_context.output_parsed.name)
        cluster_scores_no_context.append(summary_response_no_context.output_parsed.confidence)
        model_jsons_no_context.append(summary_response_no_context.model_dump_json())
    return (
        cluster_names,
        cluster_names_no_context,
        cluster_scores,
        cluster_scores_no_context,
        cluster_summaries,
        cluster_summaries_no_context,
        model_jsons,
        model_jsons_no_context,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    We create a `DataFrame` of the cluster-level summaries and associated metadata for both the context-aware and no-context methods, which we will save later.
    """)
    return


@app.cell
def _(
    cluster_names,
    cluster_names_no_context,
    cluster_scores,
    cluster_scores_no_context,
    cluster_summaries,
    cluster_summaries_no_context,
    pd,
    unique_svg_clusters,
):
    final_summary_df = pd.DataFrame({
        'cluster': unique_svg_clusters, 
        'summary': cluster_summaries, 
        'name': cluster_names, 
        'score': cluster_scores
    })
    final_summary_no_context_df = pd.DataFrame({
        'cluster': unique_svg_clusters, 
        'summary': cluster_summaries_no_context, 
        'name': cluster_names_no_context, 
        'score': cluster_scores_no_context
    })
    return final_summary_df, final_summary_no_context_df


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Let's take a look at the output\:
    """)
    return


@app.cell
def _(final_summary_df):
    final_summary_df
    return


@app.cell
def _(final_summary_no_context_df):
    final_summary_no_context_df
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    We can print the individual cluster-level summaries to Markdown; we start with cluster 0\:
    """)
    return


@app.cell
def _(final_summary_df, mo):
    mo.md(final_summary_df['summary'].to_list()[0])
    return


@app.cell
def _(final_summary_no_context_df, mo):
    mo.md(final_summary_no_context_df['summary'].to_list()[0])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Next we compare the summaries for cluster 1\:
    """)
    return


@app.cell
def _(final_summary_df, mo):
    mo.md(final_summary_df['summary'].to_list()[1])
    return


@app.cell
def _(final_summary_no_context_df, mo):
    mo.md(final_summary_no_context_df['summary'].to_list()[1])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Lastly we compare the summaries for cluster 2\:
    """)
    return


@app.cell
def _(final_summary_df, mo):
    mo.md(final_summary_df['summary'].to_list()[2])
    return


@app.cell
def _(final_summary_no_context_df, mo):
    mo.md(final_summary_no_context_df['summary'].to_list()[2])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Method comparison

    Of interest to us is whether or not our method of providing functional context to the LLM offers any improvement (and if so, how much) over simply querying the LLM with the gene set with no additional context.

    We loop over our two sets of summaries, generate embeddings, and compute the cosine similarity between the two embeddings.
    """)
    return


@app.cell
def _(cluster_summaries, cluster_summaries_no_context, gpt):
    cosine_sims = []
    embeds = []
    embeds_no_context = []
    for sumy, sumy_no_context in zip(cluster_summaries, cluster_summaries_no_context):
        sumy_embed = gpt.get_embedding(text=sumy)
        sumy_no_context_embed = gpt.get_embedding(text=sumy_no_context)
        cosine_similarity = gpt.cosine_sim(a=sumy_embed, b=sumy_no_context_embed)
        cosine_sims.append(cosine_similarity)
        embeds.append(sumy_embed)
        embeds_no_context.append(sumy_no_context_embed)
    return (cosine_sims,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Overall the cosine similarities between the summaries are quite high - this indicates that another method of evaluating the differences between the two approaches is likely necessary.
    """)
    return


@app.cell
def _(cosine_sims):
    cosine_sims
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Save data

    Finally, we'll save all the data we generated.
    """)
    return


@app.cell
def _(datetime, getpass, svg_gene_ids):
    svg_gene_ids['timestamp'] = datetime.now()
    svg_gene_ids['model'] = 'gpt-4o-mini'
    svg_gene_ids['author'] = getpass.getuser()
    svg_gene_ids.to_pickle('../../../Data/svg_gene_ids.pkl')
    return


@app.cell
def _(final_summary_df, final_summary_no_context_df):
    final_summary_df.to_pickle('../../../Data/final_summary_df.pkl')
    final_summary_no_context_df.to_pickle('../../../Data/final_summary_no_context_df.pkl')
    return


@app.cell
def _(json, model_jsons, model_jsons_no_context):
    with open('../../../Data/model_jsons.json', 'w') as f:
        json.dump(model_jsons, f, indent=2)
    with open('../../../Data/model_jsons_no_context.json', 'w') as f:
        json.dump(model_jsons_no_context, f, indent=2)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Session information
    """)
    return


@app.cell
def _(session_info):
    session_info.show()
    return


if __name__ == "__main__":
    app.run()
