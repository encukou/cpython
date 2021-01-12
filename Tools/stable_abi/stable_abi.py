import fileinput
import tokenize
import traceback
import sys
import pprint
import dataclasses

from stable_abi_parser import GeneratedParser
from pegen.tokenizer import Tokenizer

def parse(readline):
    tokengen = tokenize.generate_tokens(readline)
    tokenizer = Tokenizer(tokengen)
    parser = GeneratedParser(tokenizer)
    tree = parser.start()
    if not tree:
        raise parser.make_syntax_error(fileinput.filename())
    return tree

with fileinput.input() as file:
    tree = parse(file.readline)

# Try that round-trip through the parser gives the same info

str_repr = ''.join(n.dump() for n in tree)

_split = iter(str_repr.splitlines(keepends=True))
tree2 = parse(lambda: next(_split))

pprint.pprint([dataclasses.asdict(n) for n in tree2], sort_dicts=False)

def recursive_assert_equal(a, b, context=''):
    if a != b:
        if isinstance(a, list) and isinstance(b, list):
            if len(a) != len(b):
                raise AssertionError(f'{context}: len({a}) != len({b})')
            for i, (item_a, item_b) in enumerate(zip(a, b)):
                recursive_assert_equal(item_a, item_b, f'{context}[{i}]')
        if type(a) != type(b):
            recursive_assert_equal(type(a), type(b), f'type({context})')
        if (
            dataclasses.is_dataclass(a)
            and type(a) == type(b)
            and not isinstance(a, type)
        ):
            for field in dataclasses.fields(a):
                name = field.name
                item_a = getattr(a, name)
                item_b = getattr(b, name)
                print(name, item_a, item_b)
                recursive_assert_equal(item_a, item_b, f'{context}.{name}')
        raise AssertionError(f'{context}: {a} != {b}')

recursive_assert_equal(tree, tree2, 'tree')

# General info

for node in tree:
    print(node.dump(), end='')

print('Number of tracked stable ABI items:', len(tree))
