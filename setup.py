import importlib.util

from setuptools import Extension, setup

core_ext = Extension(
    "rixa.PMIx_core",
    sources=[
        "src/rixa/bindings/Rixa_python_native.cpp",
        "src/rixa/bindings/Rixa_pmix.cpp",
    ],
    include_dirs=["include"],
    libraries=["pmix"],
    language="c++",
    extra_compile_args=["-O3"],
    py_limited_api=True,
    define_macros=[
        ("Py_LIMITED_API", "0x030A0000"),
    ],
)

cmdclass = {}
ext_modules = [core_ext]

if importlib.util.find_spec("torch"):
    from torch.utils.cpp_extension import BuildExtension, CppExtension, library_paths

    torch_lib_path = library_paths()

    torch_ext = CppExtension(
        name="rixa._rixa_torch",
        sources=[
            "src/rixa/bindings/Rixa_pmix.cpp",
            "src/rixa/bindings/Rixa_C10.cpp",
        ],
        include_dirs=["include", "src/"],
        library_dirs=torch_lib_path,
        libraries=["pmix"],
        extra_compile_args={
            "cxx": [
                "-std=c++17",
            ],
        },
    )
    ext_modules.append(torch_ext)

    cmdclass["build_ext"] = BuildExtension

setup(
    ext_modules=ext_modules,
    cmdclass=cmdclass,
    options={"bdist_wheel": {"py_limited_api": "cp310"}},
)
