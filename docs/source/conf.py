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
html_css_files = [
    "synthpop.css",
]

html_theme_options = {
    "show_nav_level": 2,
    "icon_links": [
        {
            "name": "GitHub",
            "url": "https://github.com/duonl/synthpop-py",
            "icon": "fa-brands fa-github",
        },
        {
            "name": "Email",
            "url": "mailto:synthetische.data@duo.nl",
            "icon": "fa-solid fa-envelope",
        },
    ],
    "logo": {
        "text": "synthpop-py",
        "image_light": "_static/logo_light_theme.png",
        "image_dark": "_static/logo_dark_theme.png",
    }
}

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',
    'sphinx_design',
    'myst_parser',
    'sphinxcontrib.mermaid',
    "sphinx.ext.intersphinx",
]
myst_enable_extensions = [
    "dollarmath",
    "amsmath",
    "tasklist",
    "attrs_inline",
]
autodoc_member_order = "bysource"

autodoc_typehints = "description"

autodoc_mock_imports = ['plotly', 'typing_extensions',]
source_suffix = {
    '.rst': 'restructuredtext',
    '.txt': 'markdown',
    '.md': 'markdown',
}

intersphinx_mapping = {
    "sklearn": (
        "https://scikit-learn.org/stable/",
        "https://scikit-learn.org/stable/objects.inv",
    ),
}

mermaid_init_js = """
mermaid.initialize({
    flowchart: {
        diagramPadding: 0,
        nodeSpacing: 10,
        rankSpacing: 10
    }
});
"""