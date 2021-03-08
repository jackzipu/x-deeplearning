




from collections import namedtuple

TestCase = namedtuple('TestCase', [
    'name', 'model_name',
    'url',
    'model_dir',
    'model', 'data_sets',
    'kind',
])
