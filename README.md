# About

`genesetGPT` is a Python package that aims to allow users to summarize genes and gene sets using LLMs (currently only OpenAI models are supported). The LLM is guided by functional contexts scraped from databases like [the Human Protein Atlas](https://www.proteinatlas.org), [UniProt](https://www.uniprot.org), and NCBI's [Entrez database](https://www.ncbi.nlm.nih.gov/gene/). 

# Installation 

In order to install and build `genesetGPT` we recommend using [`uv`](https://docs.astral.sh/uv/). First, clone the repository from GitHub:

```bash 
git clone https://github.com/jr-leary7/genesetGPT.git
```

Navigate to the package directory: 

```bash
cd genesetGPT 
```

Set up the virtual environment (this will automatically be added to `.gitignore`):

```bash
uv venv
```

Use the following command to install the package to your machine:

```bash
uv pip install --editable .
```

Additionally, you can build the package for deployment using `uv`:

```bash
uv build 
```

# Example notebooks

In the `notebooks/` directory there are several [`marimo` notebooks](https://marimo.io) (a drop-in replacement for Jupyter notebooks that stores everything as pure Python code) demonstrating how to use the package. To run e.g., the spatial case study notebook run the following in your terminal:

```bash
marimo edit notebooks/spatial_case_study.py
```

# Contact information

If you encounter any issues with the package or need assistance in performing your analysis, please [open an issue](https://github.com/jr-leary7/genesetGPT/issues) or reach out [via email](mailto:j.leary@ufl.edu). 