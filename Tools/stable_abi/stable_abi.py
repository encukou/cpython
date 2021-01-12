import fileinput
import tokenize
import traceback
import sys
import pprint
import dataclasses
import argparse
from pathlib import Path
import contextlib

from stable_abi_parser import GeneratedParser
from pegen.tokenizer import Tokenizer

def parse(readline):
    tokengen = tokenize.generate_tokens(readline)
    tokenizer = Tokenizer(tokengen)
    parser = GeneratedParser(tokenizer)
    abidef = parser.start()
    if not abidef:
        raise parser.make_syntax_error(fileinput.filename())
    return abidef

def check_roundtrip(abidef):
    """Ensure that round-trip through the parser gives the same info"""
    str_repr = abidef.dump()

    _split = iter(str_repr.splitlines(keepends=True))
    abidef2 = parse(lambda: next(_split))

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

    recursive_assert_equal(abidef, abidef2, 'abidef')

# General info

def print_info(abidef):
    print(abidef.dump(), end='')

    print('Number of tracked stable ABI items:', len(abidef))


def main(argv):
    argparser = argparse.ArgumentParser(description='XXX')
    argparser.add_argument(
       '--input', '-i', type=argparse.FileType('r'),
        help='file with the Stable ABI definition',
    )
    args = argparser.parse_args(argv)
    args = argparser.parse_args(argv)
    with contextlib.ExitStack() as ctx:
        if args.input is None:
            cpython_root = Path(__file__).parent.parent.parent
            default_path = cpython_root / 'Misc/stable_abi.dat'
            input_file = default_path.open(encoding='ascii')
            ctx.enter_context(input_file)
        elif args.input == '-':
            input_file = sys.stdin
        else:
            input_file = open(args.input, encoding='ascii')
            ctx.enter_context(input_file)
        abidef = parse(input_file.readline)
        check_roundtrip(abidef)
        print_info(abidef)

if __name__ == '__main__':
    main(sys.argv[1:])

