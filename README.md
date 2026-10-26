# `genesetGPT`

<!-- badges: start -->
[![language](https://img.shields.io/badge/-Python?logo=Python&logoColor=white)](https://github.com/topics/python)
[![supported versions](https://img.shields.io/badge/python-%3E%3D3.12-blue)](https://github.com/jr-leary7/genesetGPT)
![release](https://img.shields.io/github/v/release/jr-leary7/genesetGPT?color=purple)
[![license: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
<!-- badges: end -->

# Background

`genesetGPT` is a Python package that enables researchers to precisely summarize individual genes and full gene modules using LLMs. OpenAI, Anthropic, & Gemini models are currently supported using each company's API backend. The LLMs are strictly guided by functional information pulled from databases such as [the Human Protein Atlas](https://www.proteinatlas.org), [UniProt](https://www.uniprot.org), and NCBI's [Entrez database](https://www.ncbi.nlm.nih.gov/gene/), along with user-provided biological context concerning the system being studied. 

# Setup 

## Installation

In order to install and start using `genesetGPT` we recommend a [`uv`](https://docs.astral.sh/uv/)-based workflow. From here on out, we assume a Unix-based system, though the commands for a Windows system [are very similar](https://docs.astral.sh/uv/getting-started/installation/#winget). First, create and navigate to a directory that will house your analysis (we'll call it `gene-module-analysis` here, but feel free to choose your own name) like so:

```bash 
mkdir gene-module-analysis
cd gene-module-analysis
```

Next, intialize a `uv` project and create a virtual environment (named `.venv` by default), making sure to specify our required minimum Python version:

```bash
uv init --python 3.12
uv venv --python 3.12
```

Activate your virtual environment:

```bash 
source .venv/bin/activate
```

Install `genesetGPT` and its dependencies from this GitHub repository using `pip`:

```bash
uv pip install git+https://github.com/jr-leary7/genesetGPT.git
```

## API keys

In order to use `genesetGPT`, you'll need at minimum 1) an OpenAI, Anthropic, or Gemini API key (linked to a funded acount), 2) the email address associated with your [NCBI Entrez account](https://www.ncbi.nlm.nih.gov/search/), and 3) a [MIM API key](https://omim.org/api) that you've previously registered for. Optionally, you can provide an [Entrez API key](https://www.ncbi.nlm.nih.gov/datasets/docs/v2/api/api-keys/); this is free, and only serves to increase the rate limit of your database requests from 3/sec to 10/sec. We recommend storing these in the root directory your project in a plaintext file called `.env`, then loading them into Python using a combination of the `load_dotenv()` function from [the `python-dotenv` package](https://pypi.org/project/python-dotenv/) and the `os.getenv()` function. 

For example, you should format your `.env` file like this:

```
MIM_API_KEY='01234'
ANTHROPIC_API_KEY='56789'
ENTREZ_EMAIL='username@university.edu'
```

Import the necessary libraries, then set your API keys from `.env` as environment variables:

```python
import os
from dotenv import load_dotenv
load_dotenv()
```

Lastly, define your API keys as variables in your Python session:

```python
mim_key = os.getenv('MIM_API_KEY')
anthropic_key = os.getenv('ANTHROPIC_API_KEY')
```

>[!WARNING]
>Be incredibly careful not to commit your `.env` file containing your API keys to any code-hosting service e.g., GitHub. This can be accomplished by adding it to the `.gitignore` file in your project's root directory, which you should do immediately after creating it. In addition, avoid sharing a single API key between multiple users. 

## LLM providers

Each LLM-based function in `genesetGPT` takes a string argument named `provider` that specifies whether to utilize an OpenAI (`'openai'`), Anthropic (`'anthropic'`), or Gemini (`'gemini'`) model. The model type itself is specified by the `model` string argument e.g., `model='claude-haiku-5-5'`. Each provider's available model lists & token pricing can be accessed at:
    1. [OpenAI docs](https://developers.openai.com/api/docs/pricing)
    2. [Anthropic docs](https://platform.claude.com/docs/en/models/overview)
    3. [Gemini docs](https://ai.google.dev/gemini-api/docs/pricing)

# Example usage

Load `genesetGPT` and a few other necessary packages (Anthropic's Claude LLM backend is used in this example), then import an example module of 50 genes that were significantly differentially-expressed in a cluster of B cells in [the 10X Genomics PBMC3k scRNA-seq dataset](https://scanpy.scverse.org/en/stable/tutorials/basics/clustering-2017.html). See [this script](data-raw/generate_example_gene_set.py) for the pre-processing details.

```python
import anthropic
import pandas as pd
import genesetgpt as gpt
bcell_genes = gpt.load_example_gene_set()
```

Next, define variables for your API keys / account metadata as described previously.

```python
mim_key = os.getenv('MIM_API_KEY')
entrez_key = os.getenv('ENTREZ_API_KEY')
entrez_email = os.getenv('ENTREZ_EMAIL')
claude_key = os.getenv('ANTHROPIC_API_KEY')
```

Use these two helper functions to load DataFrames containing mappings between Ensembl, Entrez, HGNC symbol, & MIM IDs. 

```python
all_hs_genes = gpt.fetch_gene_table()
mim_table = gpt.fetch_mim_table()
```

Now construct a DataFrame containing per-gene summarization prompts based on annotations pulled from Entrez, HPA, UniProt, etc. 

```python
user_prompt_df = gpt.build_prompt_df(
    gene_list=bcell_genes, 
    gene_id_table=all_hs_genes, 
    mim_mapping_table=mim_table, 
    mim_api_key=mim_key, 
    entrez_email=entrez_email, 
    entrez_api_key=entrez_key
)
```

Next, initialize an LLM client using your API key.

```python
claude_client = anthropic.Anthropic(api_key=claude_key)
```

Each individual gene is then concisely summarized and confidence-scored based on the gene-level prompts.

```python
gene_sumys = gpt.summarize_individual_genes(
    user_prompt_df=user_prompt_df, 
    provider='anthropic', 
    client=claude_client,
    model='claude-haiku-5-5', 
    n_workers=4
)
```

Lastly, the entire gene module is summarized, confidence-scored, and named based on the per-gene LLM sumaries. 

```python
module_sumy = gpt.summarize_module(
    module_genes=bcell_genes, 
    gene_sumy_df=gene_sumys, 
    provider='anthropic', 
    client=claude_client, 
    model='claude-sonnet-5-5'
)
module_sumy_df = module_sumy['module_summary_df']
```

>[!NOTE]
>We recommend choosing a stronger model for this step vs. the per-gene summarization step (e.g., Claude Sonnet 5.5 for module-level, Claude Haiku 5.5 for gene-level), as module summarization necessitates a longer input length than that of individual gene summarization. 

# Vignettes

In this repository's [notebooks subdirectory](notebooks/) there are several [`marimo` notebooks](https://marimo.io) (a drop-in replacement for Jupyter notebooks that stores everything as versionable Python code) demonstrating how to run a `genesetGPT` analysis. 

>[!IMPORTANT]
>Each example notebook imports additional dependencies that are not included with the default `genesetGPT` install e.g., `scikit-learn`, `scanpy[skmisc]`, and `squidpy` for [the spatially variable gene modules case study](notebooks/spatial_case_study.py). Each `marimo` notebook, when launched, will immediately alert you as to which additional dependencies are not already installed within your virtual environment and provide instructions as to how to add them. 

For example, to load the spatially-resolved transcriptomics case study notebook, execute the below command in your terminal after activating your `uv` virtual environment:

```bash
marimo edit notebooks/spatial_case_study.py
```

# Contact information

If you encounter any issues with the package or need assistance in performing your analysis, please [open an issue](https://github.com/jr-leary7/genesetGPT/issues) or reach out [via email](mailto:j.leary@ufl.edu).
