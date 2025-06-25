import marimo

__generated_with = "0.13.15"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    # Libraries

    We begin by loading all the libraries we'll need for our analysis, including our own `genesetgpt` library.
    """
    )
    return


@app.cell
def _():
    import os
    import json
    import shutil
    import pickle
    import getpass
    import warnings
    import scanpy as sc
    import pandas as pd
    import marimo as mo
    import session_info
    import anndata as ad
    import genesetgpt as gpt
    from openai import OpenAI
    from datetime import datetime
    from functools import partial
    from dotenv import load_dotenv
    from pydantic import BaseModel
    import matplotlib.pyplot as plt
    from pandarallel import pandarallel
    from concurrent.futures import ThreadPoolExecutor
    return (
        BaseModel,
        OpenAI,
        ThreadPoolExecutor,
        gpt,
        load_dotenv,
        mo,
        os,
        pandarallel,
        partial,
        plt,
        sc,
        session_info,
        shutil,
        warnings,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    # Settings 

    We set some global settings related to package verbosity and user warnings.
    """
    )
    return


@app.cell
def _(sc, warnings):
    sc.settings.verbosity = 0
    warnings.simplefilter('ignore', category=UserWarning)
    warnings.simplefilter('ignore', category=FutureWarning)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""Here we load in our global variables from our `.env` file.""")
    return


@app.cell
def _(load_dotenv):
    load_dotenv(dotenv_path='../.env')
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    # Data 

    Here we load the [pbmc3k dataset](https://scanpy.readthedocs.io/en/stable/generated/scanpy.datasets.pbmc3k.html) from 10X Genomics.
    """
    )
    return


@app.cell
def _(sc):
    ad_pbmc = sc.datasets.pbmc3k()
    ad_pbmc.layers['counts'] = ad_pbmc.X.copy()
    ad_pbmc.raw = ad_pbmc
    return (ad_pbmc,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""Since loading the dataset using `scanpy` creates a cache directory called `data/`, we remove it so that we don't accidentally commit it to our GitHub repository.""")
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
    mo.md(
        r"""
    # Analysis 

    ## scRNA-seq proprocessing 

    We start by filtering out low-quality genes and cells.
    """
    )
    return


@app.cell
def _(ad_pbmc, sc):
    sc.pp.filter_cells(ad_pbmc, min_counts=1000)
    sc.pp.filter_genes(ad_pbmc, min_cells=5)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""Next, we identify a set of 3,000 highly variable genes (HVGs) using the `seurat_v3` method, which explicitly models the variation in the raw counts.""")
    return


@app.cell
def _(ad_pbmc, sc):
    sc.pp.highly_variable_genes(
        ad_pbmc, 
        n_top_genes=3000, 
        flavor='seurat_v3', 
        subset=False
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""Now we depth-normalize and log1p-transform the raw counts.""")
    return


@app.cell
def _(ad_pbmc, sc):
    ad_pbmc.X = sc.pp.normalize_total(ad_pbmc, target_sum=1e4, inplace=False)['X']
    sc.pp.log1p(ad_pbmc)
    ad_pbmc.layers['norm'] = ad_pbmc.X.copy()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""We scale the normalized counts, then run PCA using just the HVG set.""")
    return


@app.cell
def _(ad_pbmc, sc):
    sc.pp.scale(ad_pbmc)
    sc.tl.pca(
        ad_pbmc, 
        n_comps=50, 
        random_state=312, 
        mask_var='highly_variable'
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""Next, we generate a shared nearest neighbor (SNN) graph in PCA space, then partition the graph into clusters via the Leiden algorithm.""")
    return


@app.cell
def _(ad_pbmc, sc):
    sc.pp.neighbors(
        ad_pbmc, 
        n_neighbors=20,
        n_pcs=30,  
        use_rep='X_pca', 
        metric='cosine', 
        random_state=312
    )
    sc.tl.leiden(
        ad_pbmc, 
        resolution=0.5, 
        flavor='igraph',
        n_iterations=2, 
        random_state=312
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""Using the UMAP algorithm, we further reduce dimensionality to 2D for visualization purposes.""")
    return


@app.cell
def _(ad_pbmc, sc):
    sc.tl.umap(ad_pbmc, random_state=312)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""Plotting the UMAP embedding colored by Leiden cluster ID shows clearly-separated clusters.""")
    return


@app.cell
def _(ad_pbmc, plt, sc):
    sc.pl.embedding(
        ad_pbmc, 
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
    mo.md(r"""Next, we utilize a series of Wilcoxon rank sum tests on the normalized counts to identify putative marker genes for each cluster.""")
    return


@app.cell
def _(ad_pbmc, sc):
    sc.tl.rank_genes_groups(
        ad_pbmc, 
        groupby='leiden', 
        layer='norm', 
        use_raw=False, 
        method='wilcoxon', 
        pts=True, 
        random_state=312
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""Plotting the top 5 possible markers for each cluster shows clear patterns. For example, cluster 3 is clearly composed of classical monocytes as it is defined by differential expression of genes such as *LYZ*, *S100A9*, and *FCN1*, while cluster 2 likely contains B cells due to specific expression of genes like *CD74*, *CD79A*, and *CD79B*.""")
    return


@app.cell
def _(ad_pbmc, sc):
    sc.pl.rank_genes_groups_dotplot(
        ad_pbmc, 
        groupby='leiden', 
        standard_scale='var', 
        n_genes=5
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""We extract the table of potential marker genes, and filter our genes with adjusted *p*-values of greater than 0.05. We then generate two lists, one containing the top 10 marker genes for the likely B cell cluster, and another containing the same for another cluster likely composed of NK cells. We then concatenate the lists into another `list` object called `markers_all`.""")
    return


@app.cell
def _(ad_pbmc, sc):
    de_df = sc.get.rank_genes_groups_df(ad_pbmc, group=None).query('pvals_adj <= 0.05 & scores > 0')
    de_df.rename(columns={'group': 'leiden', 'names': 'gene'}, inplace=True)
    markers_bcell = de_df.query("leiden == '2'").sort_values(by='pvals')['gene'].to_list()[:10]
    markers_nk = de_df.query("leiden == '4'").sort_values(by='pvals')['gene'].to_list()[:10]
    markers_all = markers_bcell + markers_nk
    return (markers_all,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## LLM summarization

    In order to summarize our gene set using OpenAI's GPT models, we first need some extra data sources. We first fetch a table containing several different gene IDs for every gene in the human genome. For completeness' sake, we also add a new column to our `DataFrame` containing any known HGNC symbol aliases / previous symbols for each gene.
    """
    )
    return


@app.cell
def _(gpt):
    all_hs_genes = gpt.fetch_gene_table()
    gene_df = all_hs_genes.query('hgnc_symbol in @markers_all').copy()
    gene_df['aliases'] = gene_df.apply(
        lambda row: gpt.get_aliases(row['hgnc_symbol'])['aliases'], 
        axis=1
    )
    gene_df
    return (gene_df,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""Next, we fetch a table of all of the Mendelian Inheritance of Man (MIM) IDs and their mappings to each other type of gene ID.""")
    return


@app.cell
def _(gpt):
    mim_table = gpt.fetch_mim_table()
    mim_table
    return (mim_table,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""With the `pandarallel` package, we loop over each row of our `DataFrame` object in parallel and generate a user prompt that we'll later use to summarize each gene's function.""")
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
def _(gene_df, gpt, mim_table, os):
    gene_df['prompt_user'] = gene_df.parallel_apply(
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
    mo.md(r"""Next, we define the developer prompt; this prompt is separate from the user prompt(s) and defines the overall tone and style of the LLMs response.""")
    return


@app.cell
def _():
    prompt_dev = 'You are an experienced computational biologist with advanced knowledge of transcriptomics analyses such as single-cell RNA-seq. When generating responses, you consider the statistical, computational, and biological angles of the question at hand. Your responses are detailed without being too overly technical.'
    return (prompt_dev,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""Here we set up our OpenAI API client.""")
    return


@app.cell
def _(OpenAI, os):
    client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
    return (client,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""Using the `partial()` function and a `ThreadPoolExecutor` we process the gene-level summarization tasks in parallel with 3 workers. We then add the gene-level summaries and confidence scores to our main `DataFrame`.""")
    return


@app.cell
def _(ThreadPoolExecutor, client, gene_df, gpt, partial, prompt_dev):
    summarize_one = partial(
        gpt.summarize_genes,
        prompt_dev=prompt_dev, 
        openai_client=client,
        openai_model='gpt-4o-mini' 
    )
    user_prompts = gene_df['prompt_user'].to_list()
    with ThreadPoolExecutor(max_workers=3) as pool:
        results = list(pool.map(summarize_one, user_prompts))
    llm_summaries, llm_scores = zip(*results)
    gene_df['llm_summary'] = llm_summaries
    gene_df['llm_confidence_score'] = llm_scores
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""We set up a short data class called `GeneSetSummary` that we'll use to format the results of our gene set summarization LLM call.""")
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
    mo.md(r"""Next, we format our set of gene-level LLM summaries for further summarization at the gene set level, then pass the query to the LLM.""")
    return


@app.cell
def _(GeneSetSummary, client, gene_df, prompt_dev):
    llm_summaries_bulleted = '\n'.join(f'- {s}' for s in gene_df['llm_summary'].to_list())
    summary_prompt = f"""
    Below are brief, independent functional descriptions of genes in a set:

    {llm_summaries_bulleted}

    Please write a concise (5–7 sentence) paragraph summarizing the common function(s) of this gene set. In addition, please provide a 0-1 score estimating how confident you are in your overall annotation. Lastly, provide a short 2-5 word name for the gene set based on your annotation. If you are unconfident in your response or the gene set appears mixed or inconsistent, say so. 
    """
    summary_response = client.responses.parse(
        model='gpt-4.1-mini', 
        input=[
            {'role': 'developer', 'content': prompt_dev}, 
            {'role': 'user', 'content': summary_prompt}
        ], 
        text_format=GeneSetSummary
    )
    return (summary_response,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""Here's the output from the LLM (gene set summary, gene set name, and confidence score)\:""")
    return


@app.cell
def _(mo, summary_response):
    mo.md(summary_response.output_parsed.summary)
    return


@app.cell
def _(mo, summary_response):
    mo.md(summary_response.output_parsed.name)
    return


@app.cell
def _(summary_response):
    summary_response.output_parsed.confidence
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""For comparison's sake, we also generate a prompt and output using just the bare LLM i.e., without providing any of the functional summaries we've put together.""")
    return


@app.cell
def _(GeneSetSummary, client, markers_all, prompt_dev):
    summary_prompt_no_context = f"""
    Here is a set of genes:

    {', '.join(markers_all)}. 

    Please write a concise (5–7 sentence) paragraph summarizing the common function(s) of this gene set. In addition, please provide a 0-1 score estimating how confident you are in your overall annotation. Lastly, provide a short 2-5 word name for the gene set based on your annotation. If you are unconfident in your response or the gene set appears mixed or inconsistent, say so. 
    """
    summary_response_no_context = client.responses.parse(
        model='gpt-4.1-mini', 
        input=[
            {'role': 'developer', 'content': prompt_dev}, 
            {'role': 'user', 'content': summary_prompt_no_context}
        ], 
        text_format=GeneSetSummary
    )
    return (summary_response_no_context,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""The output from the (contextless) LLM is below. Qualitatively, the summary from the model is pretty similar to the one we generated using the functional summaries. However, it seems a little less biologically *specific* in some ways.""")
    return


@app.cell
def _(mo, summary_response_no_context):
    mo.md(summary_response_no_context.output_parsed.summary)
    return


@app.cell
def _(mo, summary_response_no_context):
    mo.md(summary_response_no_context.output_parsed.name)
    return


@app.cell
def _(summary_response_no_context):
    summary_response_no_context.output_parsed.confidence
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""In order to compare the two summaries quantitatively we first generate 1536-dimensional embeddings of each summary using the OpenAI API.""")
    return


@app.cell
def _(gpt, summary_response, summary_response_no_context):
    summary_embed = gpt.get_embedding(text=summary_response.output_parsed.summary)
    summary_no_context_embed = gpt.get_embedding(text=summary_response_no_context.output_parsed.summary)
    return summary_embed, summary_no_context_embed


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""We utilize cosine similarity to measure how similar the responses are (they are quite similar).""")
    return


@app.cell
def _(gpt, summary_embed, summary_no_context_embed):
    gpt.cosine_sim(a=summary_embed, b=summary_no_context_embed)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""# Session information""")
    return


@app.cell
def _(session_info):
    session_info.show()
    return


if __name__ == "__main__":
    app.run()
