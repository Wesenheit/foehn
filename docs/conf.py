import os, sys

sys.path.insert(0, os.path.abspath("../src"))

project = "rixa"
copyright = "2026, Mateusz Kapusta"
author = "Mateusz Kapusta"
release = "0.0.1"

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.autosummary",
    "myst_parser",
]
autosummary_generate = True
autosummary_imported_members = True

autodoc_mock_imports = ["cuda", "nvshmem", "torch"]

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

html_theme = "alabaster"
html_static_path = ["_static"]
