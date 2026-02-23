project('foehn', 'c',
  version : '0.1.0',
  default_options : ['warning_level=2'])



py = import('python').find_installation(pure: false)

py.install_sources(
  'src/foehn/__init__.py',
  'src/foehn/pytorch.py',
  subdir: 'foehn'
)
pmix_dep = dependency('pmix', required: true)
pmix_abi_inc = include_directories('subprojects/pmix-abi-pmix-standard-v5.0')
pmix_abi_dep = declare_dependency(include_directories: pmix_abi_inc)

py.extension_module(
  'PMIx_core',
  'src/bindings/core.c',
  dependencies: [pmix_dep, pmix_abi_dep],
  install: true,
  subdir: 'foehn'
)
