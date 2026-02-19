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

pmix_inc = include_directories('subprojects/pmix-abi-pmix-standard-v5.0')
py.extension_module(
  'PMIx_core',
  'src/bindings/core.c',
  dependencies: [pmix_dep],
  include_directories: pmix_inc,
  install: true,
  subdir: 'foehn'
)


prterun = find_program('prterun', required: true)
test('pmix_distributed_test', 
  prterun,
  args: [
    '-n', '2',
    '--map-by', ':OVERSUBSCRIBE', # Force it to ignore slot limits
    '--host', 'localhost',
    'uv','run',
    files('tests/test_pmix.py')
  ],
  timeout: 30
)
test('pytorch_distributed_test', 
  prterun,
  args: [
    '-n', '2',
    '--map-by', ':OVERSUBSCRIBE', # Force it to ignore slot limits
    '--host', 'localhost',
    'uv','run',
    files('tests/test_pytorch_init.py')
  ],
  timeout: 30
)
