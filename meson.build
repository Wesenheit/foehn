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

py.extension_module(
  'PMIx_core',
  'src/bindings/core.c',
  dependencies: [pmix_dep,],
  install: true,
  subdir: 'foehn'
)
