import argparse
import sys
from typing import Any, Callable, Iterator
import dataclasses

import railroad

from pegen.build import build_parser
import pegen.grammar

argparser = argparse.ArgumentParser(
    prog="pegen", description="Pretty print the AST for a given PEG grammar"
)
argparser.add_argument("filename", help="Grammar description")

class SkipChoice(Exception):
    """Raised to hide an entire choice"""

@dataclasses.dataclass
class Container:
    items: list

    def __iter__(self):
        yield from self.items

class Choice(Container):
    def as_railroad(self):
        return railroad.Choice(0, *(n.as_railroad() for n in self))

class Sequence(Container):
    def as_railroad(self):
        return railroad.Sequence(*(n.as_railroad() for n in self))

@dataclasses.dataclass
class Decorated:
    child: object

class Optional(Decorated):
    def as_railroad(self):
        return railroad.Optional(self.child.as_railroad())

class ZeroOrMore(Decorated):
    def as_railroad(self):
        return railroad.ZeroOrMore(self.child.as_railroad())

class OneOrMore(Decorated):
    def as_railroad(self):
        return railroad.OneOrMore(self.child.as_railroad())

@dataclasses.dataclass
class Leaf:
    value: str

class Nonterminal(Leaf):
    def as_railroad(self):
        return railroad.NonTerminal(self.value)

class Terminal(Leaf):
    def as_railroad(self):
        return railroad.Terminal(self.value)

@dataclasses.dataclass
class Gather:
    item: object
    separator: object

    def as_railroad(self):
        return railroad.OneOrMore(self.item.as_railroad(), self.separator.as_railroad())

def convert_grammar(node):
    rules = {}
    for child in node:
        if not child.name.startswith('invalid_'):
            result = convert_node(child.rhs)
            if result is not None:
                result._node = child
                rules[child.name] = result
        #if child.name == 'global_stmt':
        #    break
    return rules

def make_choice(nodes):
    alternatives = []
    for node in nodes:
        try:
            alt = convert_node(node)
        except SkipChoice:
            continue
        if isinstance(alt, Choice):
            alternatives.extend(alt)
        elif alt is not None:
            alternatives.append(alt)
    if len(alternatives) == 0:
        return None
    if len(alternatives) == 1:
        return alternatives[0]
    return Choice(alternatives)

def make_sequence(nodes):
    items = []
    for node in nodes:
        item = convert_node(node)
        if isinstance(item, Sequence):
            items.extend(item)
        elif item is not None:
            items.append(item)
    if len(items) == 0:
        return None
    if len(items) == 1:
        return items[0]
    return Sequence(items)

def convert_node(node):
    print(type(node), node)
    match node:
        case pegen.grammar.Alt():
            return make_sequence(node.items)
        case pegen.grammar.Rhs():
            return make_choice(node.alts)
        case pegen.grammar.NamedItem():
            return convert_node(node.item)
        case pegen.grammar.Opt():
            result = convert_node(node.node)
            if result is None:
                return None
            return Optional(result)
        case pegen.grammar.Repeat0():
            result = convert_node(node.node)
            if result is None:
                return None
            return ZeroOrMore(result)
        case pegen.grammar.Repeat1():
            result = convert_node(node.node)
            if result is None:
                return None
            return OneOrMore(result)
        case pegen.grammar.NameLeaf():
            if node.value.startswith('invalid_'):
                raise SkipChoice()
            return Nonterminal(node.value)
        case pegen.grammar.StringLeaf():
            return Terminal(node.value)
        case pegen.grammar.PositiveLookahead():
            return None
        case pegen.grammar.NegativeLookahead():
            return None
        case pegen.grammar.Cut():
            return None
        case pegen.grammar.Gather():
            result = convert_node(node.node)
            sep = convert_node(node.separator)
            if result is None:
                if sep is None:
                    return None
                else:
                    return OneOrMore(sep)
            else:
                if sep is None:
                    return OneOrMore(result)
                else:
                    return Gather(result, sep)
        case pegen.grammar.Group():
            return make_sequence(node)
        case pegen.grammar.Forced():
            return convert_node(node.node)
        case _:
            raise TypeError(type(node))

def generate_html(data):
    with open('output.html', 'w') as file:
        file.write("<html><head><style>")
        file.write(railroad.DEFAULT_STYLE)
        file.write("</style><body>")
        for name, rule in data.items():
            file.write(f"<div><pre><code><b>{name}</b>: {rule._node.rhs}</code></pre><div>")
            railroad.Diagram(rule.as_railroad()).writeSvg(file.write)
            print(name)
            print(repr(rule._node))
            print(rule)
        file.write("</body></html>")


def main() -> None:
    args = argparser.parse_args()

    try:
        grammar, parser, tokenizer = build_parser(args.filename)
    except Exception as err:
        print("ERROR: Failed to parse grammar file", file=sys.stderr)
        sys.exit(1)

    data = convert_grammar(grammar)
    generate_html(data)


if __name__ == "__main__":
    main()
