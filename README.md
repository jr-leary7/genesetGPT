# About

`genesetGPT` is a Python package that aims to allow users to summarize genes and gene sets using LLMs (currently only OpenAI models are supported). The LLM is guided by functional contexts scraped from databases like [the Human Protein Atlas](https://www.proteinatlas.org), [UniProt](https://www.uniprot.org), and NCBI's [Entrez database](https://www.ncbi.nlm.nih.gov/gene/). 

# Installation 

In order to install and use `genesetGPT` we recommend using a [`uv`](https://docs.astral.sh/uv/)-based workflow. From here on out, we assume a Unix-based system, though the commands for a Windows system are very similar. First, make and navigate to a directory that will contain analysis (we'll call it `gene-set-analysis` here, but feel free to choose your own name) like so:

```bash 
mkdir gene-set-analysis
cd gene-set-analysis
```

Next, intialize your `uv` project and create a virtual environment (named `.venv` by default), making sure to use our recommended Python version:

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

# Example notebooks

In this repository's `notebooks/` subdirectory there are several [`marimo` notebooks](https://marimo.io) (a drop-in replacement for Jupyter notebooks that stores everything as pure Python code) demonstrating how to use the package. 

>[!IMPORTANT]
>Each example notebook imports non-default dependencies that are not included with the `genesetGPT` install e.g., `scanpy`, `squidpy`, and `scikit-learn` for `notebooks/spatial_case_study.py`. Each `marimo` notebook, when launched, will immediately alert you as to which dependencies are not installed in your virtual environment, and provide instructions as to how to add them. 

To run e.g., the spatially-resolved transcriptomics case study notebook run the following in your terminal (with your virtual environment activated):

```bash
marimo edit notebooks/spatial_case_study.py
```

# Contact information

If you encounter any issues with the package or need assistance in performing your analysis, please [open an issue](https://github.com/jr-leary7/genesetGPT/issues) or reach out [via email](mailto:j.leary@ufl.edu).
