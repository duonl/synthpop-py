# Configuration file for the Sphinx documentation builder.

# -- Project information -----------------------------------------------------

project = 'Synthpop'
copyright = '2026, Betty, Annelies, Erina, Cas'
author = 'Betty, Annelies, Erina, Cas'
release = '0.1.0'

# -- General configuration ---------------------------------------------------

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',
    'sphinx_design',
    'myst_parser',
    'sphinxcontrib.mermaid',
    'sphinx.ext.intersphinx',
]

templates_path = ['_templates']
exclude_patterns = []

# -- MyST configuration ------------------------------------------------------

myst_enable_extensions = [
    'dollarmath',
    'amsmath',
    'tasklist',
    'attrs_inline',
]

# -- Autodoc configuration ---------------------------------------------------

autodoc_member_order = 'bysource'
autodoc_typehints = 'description'
autodoc_mock_imports = ['plotly', 'typing_extensions']

# -- Source files ------------------------------------------------------------

source_suffix = {
    '.rst': 'restructuredtext',
    '.txt': 'markdown',
    '.md': 'markdown',
}

# -- Intersphinx -------------------------------------------------------------

intersphinx_mapping = {
    'sklearn': (
        'https://scikit-learn.org/stable/',
        'https://scikit-learn.org/stable/objects.inv',
    ),
}

# -- HTML output -------------------------------------------------------------

html_theme = 'pydata_sphinx_theme'

html_static_path = ['_static']

html_css_files = [
    'synthpop.css',
]

html_js_files = [
    'synthpop.js',
]

html_theme_options = {
    'show_nav_level': 2,
    'icon_links': [
        {
            'name': 'GitHub',
            'url': 'https://github.com/duonl/synthpop-py',
            'icon': 'fa-brands fa-github',
        },
        {
            'name': 'Email',
            'url': 'mailto:synthetische.data@duo.nl',
            'icon': 'fa-solid fa-envelope',
        },
    ],
    'logo': {
        'text': 'synthpop-py',
        'image_light': '_static/logo_light_theme.png',
        'image_dark': '_static/logo_dark_theme.png',
    },
}