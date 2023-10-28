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

railroad.INTERNAL_ALIGNMENT = 'left'

class SkipChoice(Exception):
    """Raised to hide an entire choice"""

class Data:
    simplified = False
    def simplify(self):
        return self

@dataclasses.dataclass
class Container(Data):
    items: list

    def __iter__(self):
        yield from self.items

    def simplify(self):
        self.items = [i.simplify() for i in self.items]
        if len(self.items) == 0:
            return Sequence([])
        return super().simplify()

class Choice(Container):
    def as_railroad(self):
        return railroad.Choice(0, *(n.as_railroad() for n in self))

    def simplify(self):
        if len(self.items) == 1:
            return self.items[0].simplify()
        new_items = []
        optional = False
        for item in self:
            item = item.simplify()
            match item:
                case Sequence([Choice()]):
                    new_items.extend(item)
                case Sequence():
                    if new_items:
                        adj = simplify_adjacent_choices(new_items[-1], item)
                        if adj is not None:
                            new_items[-1:] = adj
                            continue
                    new_items.append(item)
                case Choice(child_items):
                    new_items.extend(child_items)
                case _:
                    raise ValueError((self, item, type(item)))
        if new_items != self.items:
            if optional:
                return Optional(Choice(new_items)).simplify()
            return Choice(new_items).simplify()
        if len(self.items) == 1:
            return self.items[0]
        return super().simplify()

    def __str__(self):
        return '(C:' + ' | '.join(str(x) for x in self) + ')'

def simplify_adjacent_choices(*choices):
    match choices:
        case a, b if a == b:
            return [a]
        case Sequence([*p, a]), Sequence([*q, b]) if a == b:
            # Common tail element
            return [Sequence([
                Choice([Sequence(p), Sequence(q)]).simplify(),
                a,
            ])]
        case Sequence([a, *p]), Sequence([b, *q]) if a == b:
            # Common head element
            return [Sequence([
                a,
                Choice([Sequence(p), Sequence(q)]).simplify(),
            ])]
        case a, Sequence([Gather(b, _), Optional()]) as x if a == b:
            # First arm is redundant
            # XXX This can reorder the choices
            return [x]
        case a, Sequence([Gather(Choice([b, *_]), _), Optional(_)]) as x if a == b:
            # First arm is redundant
            # XXX This can reorder the choices
            return [x]
        case Terminal("'.'") as dot, Terminal("'...'"):
            return [dot]

class Sequence(Container):
    def as_railroad(self):
        if not self.items:
            return railroad.Skip()
        else:
            return railroad.Sequence(*(n.as_railroad() for n in self))

    def simplify(self):
        new_items = []
        for item in self:
            match item:
                case Sequence():
                    new_items.extend(item)
                case _:
                    if new_items:
                        adj = simplify_adjacent_items(new_items[-1], item)
                        if adj:
                            new_items[-1:] = adj
                            continue
                    new_items.append(item.simplify())
        if new_items != self.items:
            return Sequence(new_items).simplify()
        return super().simplify()

    def __str__(self):
        return '(S:' + ' '.join(str(x) for x in self) + ')'

def simplify_adjacent_items(*items):
    pass

@dataclasses.dataclass
class Decorated(Data):
    child: object

    def simplify(self):
        match self.child:
            case Sequence([]):
                return self.child
        new = self.child.simplify()
        if new != self.child:
            return type(self)(new)
        return super().simplify()

class Optional(Decorated):
    def as_railroad(self):
        return railroad.Optional(self.child.as_railroad())

    def simplify(self):
        match self.child:
            case Optional():
                return self.child
        return super().simplify()

    def __str__(self):
        return f'{self.child}?'

class Repeated(Decorated):
    def as_railroad(self):
        return railroad.OneOrMore(self.child.as_railroad())

    def simplify(self):
        match self.child:
            case Optional():
                return Optional(Repeated(self.child))
            case Repeated():
                return self.child
        return super().simplify()

    def __str__(self):
        return f'{self.child}+'

@dataclasses.dataclass
class Leaf(Data):
    value: str

class Nonterminal(Leaf):
    def as_railroad(self):
        return railroad.NonTerminal(self.value)

    def __str__(self):
        return self.value

    def simplify(self):
        if self.value == 'TYPE_COMMENT':
            return Sequence([])
        return super().simplify()

class Terminal(Leaf):
    def as_railroad(self):
        value = self.value
        if value[0] in {'"', "'"} and not value[1:-1].isidentifier():
            value = value[1:-1]
        return railroad.Terminal(value, cls="token")

    def __str__(self):
        return self.value

@dataclasses.dataclass
class Gather(Data):
    item: object
    separator: object

    def as_railroad(self):
        return railroad.OneOrMore(self.item.as_railroad(), self.separator.as_railroad())

    def __str__(self):
        return f'{self.separator}.{self.item}'

    def simplify(self):
        new_item = self.item.simplify()
        new_separator = self.separator.simplify()
        match new_item, new_separator:
            case Sequence([]), Sequence([]):
                return new_item
            case _, Sequence([]):
                return Repeated(new_item)
            case Sequence([]), _:
                return Optional(Repeated(new_separator))
            case _:
                if new_item != self.item or new_separator != self.separator:
                    return Gather(new_item, new_separator)
        return self

def make_choice(nodes):
    alternatives = []
    for node in nodes:
        try:
            alternatives.append(convert_node(node))
        except SkipChoice:
            continue
    return Choice(alternatives)

def convert_node(node):
    match node:
        case pegen.grammar.Rule():
            return convert_node(node.rhs)
        case pegen.grammar.Alt():
            return Sequence([convert_node(n) for n in node.items])
        case pegen.grammar.Rhs():
            return make_choice(node.alts)
        case pegen.grammar.NamedItem():
            return convert_node(node.item)
        case pegen.grammar.Opt():
            return Optional(convert_node(node.node))
        case pegen.grammar.Repeat0():
            return Optional(Repeated(convert_node(node.node)))
        case pegen.grammar.Repeat1():
            return Repeated(convert_node(node.node))
        case pegen.grammar.NameLeaf():
            if node.value.startswith('invalid_'):
                raise SkipChoice()
            return Nonterminal(node.value)
        case pegen.grammar.StringLeaf():
            return Terminal(node.value)
        case pegen.grammar.PositiveLookahead():
            return Sequence([])
        case pegen.grammar.NegativeLookahead():
            return Sequence([])
        case pegen.grammar.Cut():
            return Sequence([])
        case pegen.grammar.Gather():
            result = convert_node(node.node)
            sep = convert_node(node.separator)
            return Gather(result, sep)
        case pegen.grammar.Group():
            return Sequence([convert_node(n) for n in node])
        case pegen.grammar.Forced():
            return convert_node(node.node)
        case _:
            raise TypeError(type(node))

def generate_html(grammar):
    with open('output.html', 'w') as file:
        file.write("<html><head><style>")
        file.write(railroad.DEFAULT_STYLE)
        file.write("</style><body>")
        for node in grammar:
            name = node.name
            if name.startswith('invalid_'):
                continue
            print()
            print(name)
            print('-', repr(node))
            rule = convert_node(node)
            print('Got', rule)
            while (new := rule.simplify()) != rule:
                print('Simplified to', repr(new))
                rule = new
            match rule:
                case Sequence([]):
                    print('No output')
                    pass
                case _:
                    file.write(f"<div><pre><code><b>{name}</b>: {node.rhs}</code></pre><div>")
                    railroad.Diagram(rule.as_railroad()).writeSvg(file.write)
            #if name == 'import_from':
            #    break
        file.write("</body></html>")


def main() -> None:
    args = argparser.parse_args()

    try:
        grammar, parser, tokenizer = build_parser(args.filename)
    except Exception as err:
        print("ERROR: Failed to parse grammar file", file=sys.stderr)
        sys.exit(1)

    generate_html(grammar)


if __name__ == "__main__":
    main()
