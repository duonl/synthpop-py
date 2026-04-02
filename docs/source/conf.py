# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'Synthpop'
copyright = '2025, Betty, Annelies, Erina'
author = 'Betty, Annelies, Erina'
release = '0.1.0'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = []

templates_path = ['_templates']
exclude_patterns = []



# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

#html_theme = 'alabaster'
html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']
html_js_files = [
    "mermaid-zoom.js",
    "https://cdnjs.cloudflare.com/ajax/libs/svg-pan-zoom/3.6.1/svg-pan-zoom.min.js",
]

extensions = [
    'sphinx.ext.autodoc',
    'myst_parser',
    "sphinxcontrib.mermaid",
]
myst_enable_extensions = ["dollarmath", "amsmath","tasklist"]
autodoc_typehints = "both"

autodoc_mock_imports = ['matplotlib', 'typing_extensions',]
source_suffix = {
    '.rst': 'restructuredtext',
    '.txt': 'markdown',
    '.md': 'markdown',
}

