# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information
import os
import sys

from git import Repo
from sphinx.builders.html import StandaloneHTMLBuilder

sys.path.insert(0, os.path.abspath("../../experiment_design/"))

project = "experiment-design"
copyright = "2025, Can Bogoclu"
author = "Can Bogoclu"
release = "0.1.0"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.doctest",
    "sphinx.ext.autodoc",
    "sphinx_autodoc_typehints",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.githubpages",
]


templates_path = ["_templates"]
exclude_patterns = []


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "sphinx_rtd_theme"
html_static_path = ["_static"]

html_theme_options = {
    "logo_only": False,
    "prev_next_buttons_location": "bottom",
    "style_external_links": False,
    "vcs_pageview_mode": "",
    "style_nav_header_background": "#2980B9",
    "flyout_display": "hidden",
    "version_selector": True,
    "language_selector": True,
    # Toc options
    "collapse_navigation": True,
    "sticky_navigation": True,
    "navigation_depth": 3,
    "includehidden": True,
    "titles_only": False,
}

StandaloneHTMLBuilder.supported_image_types = [
    "image/svg+xml",
    "image/gif",
    "image/png",
    "image/jpeg",
]

# SET CURRENT_VERSION


if "REPO_NAME" in os.environ:
    REPO_NAME = os.environ["REPO_NAME"]
else:
    REPO_NAME = ""

try:
    html_context
except NameError:
    html_context = dict()
html_context["display_lower_left"] = True

repo = Repo(search_parent_directories=True)

if "current_version" in os.environ:
    # get the current_version env var set by buildDocs.sh
    current_version = os.environ["current_version"]
else:
    # the user is probably doing `make html`
    # set this build's current version by looking at the branch
    current_version = repo.active_branch.name

# tell the theme which version we're currently on ('current_version' affects
# the lower-left rtd menu and 'version' affects the logo-area version)
html_context["current_version"] = current_version
html_context["version"] = current_version

# POPULATE LINKS TO OTHER VERSIONS
html_context["versions"] = list()

versions = [branch.name for branch in repo.branches]
for version in versions:
    html_context["versions"].append((version, "/" + REPO_NAME + "/" + version + "/"))

html_context["display_github"] = True
html_context["github_user"] = "canbooo"
html_context["github_repo"] = "experiment-design"
html_context["github_version"] = "master/docs/"
