# conf.py

# Import necessary modules
import os
import sys
sys.path.insert(0, os.path.abspath('../src'))  # Adjust path to point to your project source code

# Project information
project = 'DataLib'
author = 'Oussama ELAYEB'
release = '1.0'
master_doc = 'index'
# Sphinx Extensions
extensions = [
    'sphinx.ext.autodoc',     # Generate documentation from docstrings
    'sphinx.ext.napoleon',    # Support for NumPy/Google-style docstrings
    'sphinx.ext.viewcode',    # Add links to source code 
]

# Paths
templates_path = ['_templates']
exclude_patterns = []

# HTML Theme
html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']

# Add Markdown file support
source_suffix = {
    '.rst': 'restructuredtext',
}