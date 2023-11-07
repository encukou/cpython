import argparse
import sys
from typing import Any, Callable, Iterator
import dataclasses
from collections import Counter

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

class Item:
    simplified = False
    def simplified(self):
        return self

@dataclasses.dataclass
class Container(Item):
    items: list

    def __iter__(self):
        yield from self.items

    def simplified(self):
        self.items = [i.simplified() for i in self.items]
        if len(self.items) == 0:
            return Sequence([])
        return super().simplified()

    def walk(self):
        yield self
        for item in self:
            yield from item.walk()

    def inlined(self, rule):
        return type(self)([item.inlined(rule) for item in self])

class Choice(Container):
    def as_railroad(self):
        return railroad.Choice(0, *(n.as_railroad() for n in self))

    def simplified(self):
        if len(self.items) == 1:
            return self.items[0].simplified()
        new_items = []
        optional = False
        for item in self:
            item = item.simplified()
            match item:
                case Sequence([Choice()]):
                    new_items.extend(item)
                case Sequence():
                    if new_items:
                        adj = simplified_adjacent_choices(new_items[-1], item)
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
                return Optional(Choice(new_items)).simplified()
            return Choice(new_items).simplified()
        if len(self.items) == 1:
            return self.items[0]
        return super().simplified()

    def __str__(self):
        return '(C:' + ' | '.join(str(x) for x in self) + ')'

def simplified_adjacent_choices(*choices):
    match choices:
        case a, b if a == b:
            return [a]
        case Sequence([*p, a]), Sequence([*q, b]) if a == b:
            # Common tail element
            return [Sequence([
                Choice([Sequence(p), Sequence(q)]).simplified(),
                a,
            ])]
        case Sequence([a, *p]), Sequence([b, *q]) if a == b:
            # Common head element
            return [Sequence([
                a,
                Choice([Sequence(p), Sequence(q)]).simplified(),
            ])]
        case Sequence([a]), Sequence([
            Gather(
                Sequence([Choice([Sequence([b]), *_])]),
                _,
            ),
            Choice([*_, Sequence([])])
        ]) if a == b:
            # First arm is redundant.
            # XXX This can reorder the choices.
            return [choices[-1]]
        case Terminal("'.'") as dot, Terminal("'...'"):
            return [dot]

class Sequence(Container):
    def as_railroad(self):
        if not self.items:
            return railroad.Skip()
        else:
            return railroad.Sequence(*(n.as_railroad() for n in self))

    def simplified(self):
        new_items = []
        for item in self:
            match item:
                case Sequence():
                    new_items.extend(item)
                case _:
                    if new_items:
                        adj = simplified_adjacent_items(new_items[-1], item)
                        if adj:
                            new_items[-1:] = adj
                            continue
                    new_items.append(item.simplified())
        if new_items != self.items:
            return Sequence(new_items).simplified()
        return super().simplified()

    def __str__(self):
        return '(S:' + ' '.join(str(x) for x in self) + ')'

def simplified_adjacent_items(*items):
    pass

@dataclasses.dataclass
class Decorated(Item):
    child: object

    def simplified(self):
        match self.child:
            case Sequence([]):
                return self.child
        new = self.child.simplified()
        if new != self.child:
            return type(self)(new)
        return super().simplified()

    def walk(self):
        yield self
        yield from self.child.walk()

    def inlined(self, rule):
        return type(self)(self.child.inlined(rule))

class Repeated(Decorated):
    def as_railroad(self):
        return railroad.OneOrMore(self.child.as_railroad())

    def simplified(self):
        match self.child:
            #case Optional():
            #    return Optional(Repeated(self.child))
            case Repeated():
                return self.child
        return super().simplified()

    def __str__(self):
        return f'{self.child}+'

@dataclasses.dataclass
class Leaf(Item):
    value: str

    def walk(self):
        yield self

    def inlined(self, rule):
        return self

class Nonterminal(Leaf):
    def as_railroad(self):
        return railroad.NonTerminal(self.value)

    def __str__(self):
        return self.value

    def simplified(self):
        if self.value == 'TYPE_COMMENT':
            return Sequence([])
        return super().simplified()

    def inlined(self, rule):
        if self.value == rule.name:
            return rule.item
        else:
            return self

class Terminal(Leaf):
    def as_railroad(self):
        value = self.value
        if value[0] in {'"', "'"} and not value[1:-1].isidentifier():
            value = value[1:-1]
        return railroad.Terminal(value, cls="token")

    def __str__(self):
        return self.value

@dataclasses.dataclass
class Gather(Item):
    item: object
    separator: object

    def as_railroad(self):
        return railroad.OneOrMore(self.item.as_railroad(), self.separator.as_railroad())

    def __str__(self):
        return f'{self.separator}.{self.item}'

    def simplified(self):
        new_item = self.item.simplified()
        new_separator = self.separator.simplified()
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

    def walk(self):
        yield self
        yield from self.item.walk()
        yield from self.separator.walk()

    def inlined(self, rule):
        return Gather(self.item.inlined(rule), self.separator.inlined(rule))

class Rule:
    item: Item
    node: pegen.grammar.Rule
    name: str
    usages: Counter

    def __init__(self, node):
        self.item = convert_node(node)
        self.node = node
        self.name = node.name
        self.usages = Counter()

    def simplify(self):
        while (new := self.item.simplified()) != self.item:
            print('Simplified to', repr(new))
            self.item = new

    def inline(self, rule):
        self.item = self.item.inlined(rule)

    def as_railroad(self):
        return self.item.as_railroad()

    def reset_usages(self):
        self.usages = Counter()

    def fill_usages(self, rules):
        for item in self.item.walk():
            match item:
                case Nonterminal(name) if name in rules:
                    rules[name].usages[self.name] += 1

    def __str__(self):
        return f'{self.name}::= {self.item}'

def make_choice(nodes):
    alternatives = []
    for node in nodes:
        try:
            alternatives.append(convert_node(node))
        except SkipChoice:
            continue
    return Choice(alternatives)

def make_opt(item):
    return Choice([Sequence([item]), Sequence([])])

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
            return make_opt(convert_node(node.node))
        case pegen.grammar.Repeat0():
            return make_opt(Repeated(convert_node(node.node)))
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

def generate_html(rules):
    with open('output.html', 'w') as file:
        file.write("<html><head><style>")
        file.write("""
            svg.railroad-diagram path {
                stroke-width:3;
                stroke:black;
                fill: transparent;
            }
            svg.railroad-diagram text {
                font:bold 14px monospace;
                text-anchor:middle;
            }
            svg.railroad-diagram text.label{
                text-anchor:start;
            }
            svg.railroad-diagram text.comment{
                font:italic 12px monospace;
            }
            svg.railroad-diagram rect{
                stroke-width:3;
                stroke:black;
                fill: #bbddff;
            }
            svg.railroad-diagram rect.group-box {
                stroke: gray;
                stroke-dasharray: 10 5;
                fill: none;
            }
            svg.railroad-diagram .token rect {
                fill: #ffffaa
            }
        """)
        file.write("</style><body>")
        for name, rule in rules.items():
            print()
            print(name)
            print('Node:', repr(rule.node))
            print('Rule:', rule)
            match rule:
                case Sequence([]):
                    print('No output')
                    pass
                case _:
                    file.write(f"<div><pre><code><b>{name}</b>: {rule.node.rhs}</code></pre><div>")
                    if rule.usages:
                        usages = ', '.join(
                            f'{name} ({count})'
                            for name, count in rule.usages.most_common()
                        )
                        file.write(f"<div>Used in: {usages}</div>")
                    else:
                        file.write(f"<div>Unused!</div>")
                    railroad.Diagram(rule.as_railroad()).writeSvg(file.write)
            #if name == 'slices':
            #    break
        file.write("</body></html>")


def main() -> None:
    args = argparser.parse_args()

    try:
        grammar, parser, tokenizer = build_parser(args.filename)
    except Exception as err:
        print("ERROR: Failed to parse grammar file", file=sys.stderr)
        sys.exit(1)

    rules = {
        node.name: Rule(node) for node in grammar
        if not node.name.startswith('invalid_')
    }
    for rule in rules.values():
        rule.simplify()
        rule.reset_usages()
    for rule in rules.values():
        rule.fill_usages(rules)
    for rule in list(rules.values()):
        if len(rule.usages) == 1:
            parent_name = rule.usages.most_common()[0][0]
            try:
                parent_rule = rules[parent_name]
            except KeyError:
                pass
            else:
                parent_rule.inline(rule)
                del rules[rule.name]

    generate_html(rules)


if __name__ == "__main__":
    main()
