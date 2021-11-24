# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
import os
import sys

import sphinx
import sphinx_gallery
from matplotlib.sphinxext import plot_directive

sys.path.insert(0, os.path.abspath('../../'))


# -- Project information -----------------------------------------------------

project = 'pyLFI'
copyright = '2021, Nicolai Haug'
author = 'Nicolai Haug'

# Get version
about = {}
with open(os.path.join("..", "..", "pylfi", "__version__.py")) as f:
    exec(f.read(), about)

# The version info for the project you're documenting, acts as replacement for
# |version|, also used in various other places throughout the built documents.
#
version = about['__version__']


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
#
extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.doctest',
    'sphinx.ext.autosummary',
    'sphinx.ext.extlinks',
    'sphinx.ext.intersphinx',
    'sphinx.ext.inheritance_diagram',
    'sphinx.ext.mathjax',
    'sphinx.ext.viewcode',
    'numpydoc',
    'matplotlib.sphinxext.plot_directive',
    # 'sphinx_gallery.gen_gallery',
]

# Add autodoc mock imports
#
"""
autodoc_mock_imports = [
    'numpy',
    'matplotlib',
    'scipy',
    'seaborn',
    'pandas',
    'sklearn',
]
"""
# -- Configure extensions  ----------------------------------------------------

# -- Numpy extensions
#
#numpydoc_use_plots = True

# this is needed for some reason...
# see https://github.com/numpy/numpydoc/issues/69
#
numpydoc_class_members_toctree = False
numpydoc_attributes_as_param_list = False

# -- matplotlib plot directive
#
plot_include_source = True
plot_formats = [("png", 90)]
plot_html_show_formats = False
plot_html_show_source_link = False
# plot_pre_code = """import numpy as np
# import pandas as pd"""

# -- Sphinx gallery
"""
sphinx_gallery_conf = {
    'examples_dirs': ['../examples', '../tutorials'],
    'gallery_dirs': ['auto_examples']
}
"""

# -- Autodoc
#
autodoc_member_order = 'bysource'

autodoc_default_options = {
    'members': True,
    'special-members': False,
    'inherited-members': True
}

# -- Intersphinx
#
intersphinx_mapping = {
    "python": (
        "https://docs.python.org/3/",
        (None, "http://www.astropy.org/astropy-data/intersphinx/python3.inv"),
    ),
    "numpy": (
        "https://numpy.org/doc/stable/",
        (None, "http://www.astropy.org/astropy-data/intersphinx/numpy.inv"),
    ),
    "scipy": (
        "https://docs.scipy.org/doc/scipy/reference/",
        (None, "http://www.astropy.org/astropy-data/intersphinx/scipy.inv"),
    ),
    "matplotlib": (
        "https://matplotlib.org/",
        (None, "http://www.astropy.org/astropy-data/intersphinx/matplotlib.inv"),
    ),
    'pandas': (
        'https://pandas.pydata.org/pandas-docs/stable/',
        (None),
    ),
    'seaborn': (
        'https://seaborn.pydata.org/',
        (None),
    ),
}


# -- Inheritance diagram
#
inheritance_node_attrs = dict(
    shape="ellipse", fontsize=12, color="orange", style="filled"
)

viewcode_import = True

# -- Autosummary
# generate autosummary even if no references
#
autosummary_generate = True
autosummary_imported_members = True

# -- Additional configuration ------------------------------------------------

# Add any paths that contain templates here, relative to this directory.
#
templates_path = ['_templates']

# The suffix(es) of source filenames.
#
source_suffix = '.rst'

# The master toctree document.
#
master_doc = 'index'

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
#
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

# The reST default role to use for all documents.
#
default_role = 'literal'

# If true, '()' will be appended to :func: etc. cross-reference text.
#
add_function_parentheses = True

# The name of the Pygments (syntax highlighting) style to use.
#
pygments_style = 'sphinx'

default_role = 'obj'

# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = "pydata_sphinx_theme"

htmlhelp_basename = 'pylfidoc'

html_context = {
    "display_github": True,
    "github_host": "github.com",
    "gitlab_user": "nicolossus",
    "gitlab_repo": 'pylfi',
    "gitlab_version": "HEAD",
    "conf_py_path": "/docs/",
    "source_suffix": '.rst'
}

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
#
html_static_path = ['_static']
