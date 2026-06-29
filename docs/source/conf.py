# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'Synthpop'
copyright = '2026, Betty, Annelies, Erina, Cas'
author = 'Betty, Annelies, Erina, Cas'
release = '0.1.0'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = []

templates_path = ['_templates']
exclude_patterns = []



# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

#html_theme = 'alabaster'
html_theme = 'pydata_sphinx_theme'
html_static_path = ['_static']

html_theme_options = {
    "show_nav_level": 2,
    "icon_links": [
        {
            "name": "GitHub",
            "url": "https://github.com/duonl/synthpop-py",
            "icon": "fa-brands fa-github",
        },
    ],
    "logo": {
        "text": "Synthpop-py",
        "image_light": "_static/_static/logo_light_theme.png",
        "image_dark": "_static/_static/logo_dark_theme.png",
    }
}

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',
    'myst_parser',
    'sphinxcontrib.mermaid',
]
myst_enable_extensions = ["dollarmath", "amsmath","tasklist"]
autodoc_member_order = "bysource"

autodoc_typehints = "none"

autodoc_mock_imports = ['plotly', 'typing_extensions',]
source_suffix = {
    '.rst': 'restructuredtext',
    '.txt': 'markdown',
    '.md': 'markdown',
}