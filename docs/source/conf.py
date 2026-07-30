"""Sphinx configuration for genome_entropy documentation."""

import os
import sys
from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as package_version
from typing import Any

# Add source directory to path for autodoc
sys.path.insert(0, os.path.abspath("../../src"))

# Project information
project = "genome_entropy"
copyright = "2026, Rob Edwards"
author = "Rob Edwards"

# Get version from package metadata
try:
    release = package_version("genome_entropy")
    version = release
except PackageNotFoundError:
    release = "0.0.0"
    version = "0.0.0"

# General configuration
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.intersphinx",
    "myst_parser",
]

# Autodoc settings
autodoc_default_options = {
    "members": True,
    "member-order": "bysource",
    "special-members": "__init__",
    "undoc-members": True,
    "exclude-members": "__weakref__",
}
autodoc_typehints = "description"

# Napoleon settings (for Google/NumPy style docstrings)
napoleon_google_docstring = True
napoleon_numpy_docstring = False
napoleon_include_init_with_doc = True
napoleon_include_private_with_doc = False
napoleon_include_special_with_doc = True
napoleon_use_admonition_for_examples = False
napoleon_use_admonition_for_notes = False
napoleon_use_admonition_for_references = False
napoleon_use_ivar = True
napoleon_use_param = True
napoleon_use_rtype = True
napoleon_preprocess_types = False
napoleon_type_aliases = None
napoleon_attr_annotations = True

# MyST parser settings (for Markdown support)
myst_enable_extensions = [
    "colon_fence",
    "deflist",
    "dollarmath",
    "fieldlist",
    "html_admonition",
    "html_image",
    "linkify",
    "replacements",
    "smartquotes",
    "substitution",
    "tasklist",
]

# Source suffix
source_suffix = {
    ".rst": "restructuredtext",
    ".md": "markdown",
}

# Master document
master_doc = "index"

# Language
language = "en"

# List of patterns to exclude
exclude_patterns: list[str] = []

# HTML output options
html_theme = "sphinx_rtd_theme"
html_theme_options = {
    "navigation_depth": 4,
    "collapse_navigation": False,
    "sticky_navigation": True,
    "includehidden": True,
    "titles_only": False,
    "prev_next_buttons_location": "bottom",
    "style_external_links": True,
}

# Intersphinx mapping
intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "torch": ("https://docs.pytorch.org/docs/stable/", None),
}

# Add any paths that contain custom static files (such as style sheets)
html_css_files: list[str] = []

# Output file base name for HTML help builder
htmlhelp_basename = "genome_entropydoc"

# GitHub link
html_context = {
    "display_github": True,
    "github_user": "linsalrob",
    "github_repo": "genome_entropy",
    "github_version": "main",
    "conf_py_path": "/docs/source/",
}


def configure_builder_specific_files(app: Any) -> None:
    """Include GitHub Pages files only in browser-oriented HTML builds."""
    # EPUB also has ``format == "html"``, so select concrete HTML builders.
    if app.builder.name in {"html", "dirhtml", "singlehtml"}:
        app.config.html_extra_path = [".nojekyll"]
    else:
        app.config.html_extra_path = []


def setup(app: Any) -> dict[str, object]:
    """Configure builder-specific documentation files."""
    app.connect("builder-inited", configure_builder_specific_files)
    return {
        "parallel_read_safe": True,
        "parallel_write_safe": True,
    }
