from setuptools import setup, Extension
import importlib.util

core_ext = Extension(
    "rixa.PMIx_core",
    sources=["src/bindings/core.c", "src/bindings/rixa_pmix_store.c"],
    include_dirs=["include"],
    libraries=["pmix"],
    language="c",
    extra_compile_args=["-O3"],
)

cmdclass = {}
ext_modules = [core_ext]

if importlib.util.find_spec("torch"):
    from torch.utils.cpp_extension import CppExtension, BuildExtension
    import torch

    torch_lib_path = torch.utils.cpp_extension.library_paths()

    torch_ext = CppExtension(
        name="rixa._rixa_torch",
        sources=[
            "src/bindings/rixa_pmix_store.c",
            "src/bindings/rixa_C10.cpp",
        ],
        include_dirs=["include", "src/"],
        libraries=["pmix", "torch", "c10", "torch_python"],
        library_dirs=torch_lib_path,
        extra_compile_args={
            "cxx": [
                "-std=c++17",
            ],  # only for .cpp files
            "cc": ["-O3"],  # only for .c files
        },
        extra_link_args=[
            "-Wl,--no-as-needed",
            "-ltorch_python",
            "-Wl,--as-needed",
        ],
    )
    ext_modules.append(torch_ext)

    cmdclass["build_ext"] = BuildExtension.with_options(use_ninja=False)

setup(
    ext_modules=ext_modules,
    cmdclass=cmdclass,
)
