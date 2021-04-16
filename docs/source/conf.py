# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

import math
import os
import sys

import matplotlib
import matplotlib.pyplot as plt
import sphinx_rtd_theme
from matplotlib.sphinxext import plot_directive

# Get version
exec(open(os.path.join("..", "..", "pylfi", "_version.py")).read())

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
#import pylfi

#sys.path.insert(0, os.path.abspath(os.pardir))
#sys.path.insert(0, os.path.abspath('../sphinxext'))
sys.path.insert(0, os.path.abspath('../../'))


# -- Project information -----------------------------------------------------

project = u'pyLFI'
copyright = u'2021, Nicolai Haug'
author = u'Nicolai Haug'

# The version info for the project you're documenting, acts as replacement for
# |version| and |release|, also used in various other places throughout the
# built documents.

# The short X.Y version.
# The short X.Y version.
version = __version__
# The full version, including alpha/beta/rc tags.
release = __version__


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.

autodoc_mock_imports = ['numpy', 'numpy.random', 'matplotlib',
                        'matplotlib.pyplot', 'scipy', 'scipy.stats',
                        'scipy.optimize', 'scipy.signal', 'scipy.special',
                        'scipy.integrate', 'scipy.interpolate',
                        'seaborn', 'pandas', 'sklearn', 'sklearn.linear_model',
                        'sklearn.metrics', 'sklearn.model_selection',
                        'sklearn.neighbors', 'torch', 'coverage', 'ot']
"""
'pylfi', 'pylfi.density_estimation', 'pylfi.distances',
'pylfi.features', 'pylfi.inferences', 'pylfi.models',
'pylfi.plotting', 'pylfi.priors', 'pylfi.simulators',
'pylfi.utils', 'pylfi.journal']
"""

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinx.ext.coverage',
    'sphinx.ext.doctest',
    'sphinx.ext.extlinks',
    # 'sphinx.ext.githubpages',
    # 'sphinx.ext.imgconverter',
    'sphinx.ext.intersphinx',
    'sphinx.ext.inheritance_diagram',
    # 'sphinx.ext.linkcode',
    'sphinx.ext.mathjax',
    # 'sphinx.ext.napoleon',
    # 'sphinx-prompt',
    # 'sphinx.ext.todo',
    'sphinx.ext.viewcode',
    # 'sphinx_gallery.gen_gallery',
    # 'sphinx_issues',
    # 'add_toctree_functions',
    # "sphinx_rtd_theme",
    # 'sphinx_rtd_theme_ext_color_contrast',
    # 'sphinxcontrib.napoleon',
    'numpydoc',
    'matplotlib.sphinxext.plot_directive',
]

# Do some matplotlib config in case users have a matplotlibrc that will break
# things
# matplotlib.use('agg')
# plt.ioff()


# -----------------------------------------------------------------------------
# Numpy extensions
# -----------------------------------------------------------------------------
#numpydoc_use_plots = True


# this is needed for some reason...
# see https://github.com/numpy/numpydoc/issues/69
numpydoc_class_members_toctree = False

# -----------------------------------------------------------------------------
# Autodoc
# -----------------------------------------------------------------------------
#autodoc_default_flags = ['members']

autodoc_member_order = 'bysource'

autodoc_default_options = {
    'members': True,
    # 'undoc-members': False,
    'special-members': False,
    # 'exclude-members': '__weakref__',
    'inherited-members': True
}

autodoc_typehints = 'none'

# -----------------------------------------------------------------------------
# Intersphinx configuration
# -----------------------------------------------------------------------------
'''
intersphinx_mapping = {
    'python': ('https://docs.python.org/{.major}'.format(
        sys.version_info), None),
    'numpy': ('https://numpy.org/doc/stable', None),
    'scipy': ('https://docs.scipy.org/doc/scipy/reference', None),
    'matplotlib': ('https://matplotlib.org/', None),
    'pandas': ('https://pandas.pydata.org/pandas-docs/stable/', None),
    'joblib': ('https://joblib.readthedocs.io/en/latest/', None),
    'seaborn': ('https://seaborn.pydata.org/', None),
}
'''

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

# inheritance_diagram configuration
inheritance_node_attrs = dict(
    shape="ellipse", fontsize=12, color="orange", style="filled"
)

viewcode_import = True

# Napoleon options
'''
napoleon_google_docstring = False
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = False
napoleon_include_private_with_doc = False
napoleon_include_special_with_doc = False
napoleon_use_admonition_for_examples = True
napoleon_use_admonition_for_notes = True
napoleon_use_admonition_for_references = True
napoleon_use_ivar = False
napoleon_use_param = False
napoleon_use_rtype = True
napoleon_use_keyword = True
napoleon_custom_sections = None
#napoleon_custom_sections = [("Side Effects", "Parameters") ]
'''

# -----------------------------------------------------------------------------
# Autosummary
# -----------------------------------------------------------------------------
# generate autosummary even if no references
autosummary_generate = True
autosummary_imported_members = True

# ------------------------------------------------------------------------------
# Matplotlib plot_directive options
# ------------------------------------------------------------------------------
'''
plot_pre_code = """
import numpy as np
np.random.seed(123)
"""
plot_include_source = True
plot_formats = [('png', 96), 'pdf']
plot_html_show_formats = False
plot_html_show_source_link = False

phi = (math.sqrt(5) + 1) / 2

font_size = 13 * 72 / 96.0  # 13 px

plot_rcparams = {
    'font.size': font_size,
    'axes.titlesize': font_size,
    'axes.labelsize': font_size,
    'xtick.labelsize': font_size,
    'ytick.labelsize': font_size,
    'legend.fontsize': font_size,
    'figure.figsize': (3 * phi, 3),
    'figure.subplot.bottom': 0.2,
    'figure.subplot.left': 0.2,
    'figure.subplot.right': 0.9,
    'figure.subplot.top': 0.85,
    'figure.subplot.wspace': 0.4,
    'text.usetex': False,
}
'''

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']


# The suffix(es) of source filenames.
# You can specify multiple suffix as a list of string:
#
# source_suffix = ['.rst', '.md']
source_suffix = '.rst'

# The master toctree document.
master_doc = 'index'

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.

exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

# The reST default role (used for this markup: `text`) to use for all
# documents.
#
# default_role = None
default_role = 'literal'
#default_role = "autolink"

# If true, '()' will be appended to :func: etc. cross-reference text.
#
add_function_parentheses = True

# If true, the current module name will be prepended to all description
# unit titles (such as .. function::).
#
# add_module_names = True

# If true, sectionauthor and moduleauthor directives will be shown in the
# output. They are ignored by default.
#
# show_authors = False

# The name of the Pygments (syntax highlighting) style to use.
pygments_style = 'sphinx'

# A list of ignored prefixes for module index sorting.
# modindex_common_prefix = []

# If true, keep warnings as "system message" paragraphs in the built documents.
# keep_warnings = False

# If true, `todo` and `todoList` produce output, else they produce nothing.
# todo_include_todos = True

# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
#html_theme = 'alabaster'
#html_theme = 'nature'

htmlhelp_basename = 'pylfidoc'

#html_title = project

#html_theme = "sphinx_rtd_theme"
html_theme = "pydata_sphinx_theme"

html_context = {
    "display_github": True,
    "github_host": "github.com",
    "gitlab_user": "nicolossus",
    "gitlab_repo": 'master-thesis',
    "gitlab_version": "HEAD",
    "conf_py_path": "/docs/source/",
    "source_suffix": '.rst'
}


# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']
