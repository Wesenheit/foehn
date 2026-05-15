from setuptools import setup, Extension

ext_modules = [
    Extension(
        "rixa.PMIx_core",
        sources=[
            "src/bindings/core.c",
        ],
        include_dirs=[
            "include",
        ],
        libraries=["pmix"],
        language="c",
        extra_compile_args=["-O3"],
    ),
]

setup(ext_modules=ext_modules)
