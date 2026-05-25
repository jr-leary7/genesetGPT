`genesetGPT`

<!-- badges: start -->
[![language](https://img.shields.io/badge/-Python?logo=Python&logoColor=white)](https://github.com/topics/python)
[![supported versions](https://img.shields.io/badge/python-%3E%3D3.12-blue)](https://github.com/jr-leary7/genesetGPT)
![release](https://img.shields.io/github/v/release/jr-leary7/genesetGPT?color=purple)
[![license: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
<!-- badges: end -->

# Background

`genesetGPT` is a Python package that enables researchers to precisely summarize individual genes and larger gene sets using LLMs (currently only OpenAI models are supported, but support for Anthropic models is coming soon). The LLMs are strictly guided by functional information pulled from databases such as [the Human Protein Atlas](https://www.proteinatlas.org), [UniProt](https://www.uniprot.org), and NCBI's [Entrez database](https://www.ncbi.nlm.nih.gov/gene/), along with user-provided biological context concerning the system being studied. 

# Installation 

In order to install and start using `genesetGPT` we recommend a [`uv`](https://docs.astral.sh/uv/)-based workflow. From here on out, we assume a Unix-based system, though the commands for a Windows system [are very similar](https://docs.astral.sh/uv/getting-started/installation/#winget). First, create and navigate to a directory that will house your analysis (we'll call it `gene-set-analysis` here, but feel free to choose your own name) like so:

```bash 
mkdir gene-set-analysis
cd gene-set-analysis
```

Next, intialize your `uv` project and create a virtual environment (named `.venv` by default), making sure to use our required minimum Python version:

```bash
uv init --python 3.12
uv venv --python 3.12
```

Activate your virtual environment:

```bash 
source .venv/bin/activate
```

Now you can install `genesetGPT` and its dependencies from this GitHub repository using `pip`:

```bash
uv pip install git+https://github.com/jr-leary7/genesetGPT.git
```

# Tutorial notebooks

In this repository's `notebooks/` subdirectory there are several [`marimo` notebooks](https://marimo.io) (a drop-in replacement for Jupyter notebooks that stores everything as versionable Python code) demonstrating how to use the package. 

>[!IMPORTANT]
>Each example notebook imports dependencies that are not included with the default `genesetGPT` install e.g., `scikit-learn`, `scanpy[skmisc]`, and `squidpy` for `notebooks/spatial_case_study.py`. Each `marimo` notebook, when launched, will immediately alert you as to which notebook dependencies are not installed in your virtual environment, and provide instructions as to how to add them. 

For example, to start runinng the spatially-resolved transcriptomics case study notebook, execute the following in your terminal (with your virtual environment activated):

```bash
marimo edit notebooks/spatial_case_study.py
```

# Contact information

If you encounter any issues with the package or need assistance in performing your analysis, please [open an issue](https://github.com/jr-leary7/genesetGPT/issues) or reach out [via email](mailto:j.leary@ufl.edu).
