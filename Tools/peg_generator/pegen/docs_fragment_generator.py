import argparse
import sys
from typing import Any, Callable, Iterator
import json
from dataclasses import dataclass

from pegen.build import build_parser
from pegen.grammar import Grammar, Rule
import pegen.grammar

argparser = argparse.ArgumentParser(
    prog="pegen", description="Pretty print the AST for a given PEG grammar"
)
argparser.add_argument("grammar_filename", help="Grammar description")
argparser.add_argument("fragments_filename", help="List of rules that should be documented")
argparser.add_argument("output_filename", help="File to write the output to")

class InvalidRuleIncluded(Exception):
    """Refusal to format an `invalid_` rule"""


@dataclass
class Node:
    pass


@dataclass
class Container(Node):
    """Collection of zero or more children"""
    items: list[Node]

    def __iter__(self):
        yield from self.items


@dataclass
class Choice(Container):
    def format(self):
        option_reprs = []
        for item in self:
            try:
                option_reprs.append(item.format())
            except InvalidRuleIncluded:
                pass
        return " | ".join(option_reprs)


@dataclass
class Sequence(Container):
    def format(self):
        item_reprs = []
        for item in self:
            item_repr = item.format()
            if item_repr:
                item_reprs.append(item_repr)
        return " ".join(item_reprs)


@dataclass
class Decorator(Node):
    """Node with exactly one child"""
    item: Node

    def __iter__(self):
        yield self.item


@dataclass
class Optional(Decorator):
    def format(self):
        return '[' + self.item.format() + ']'


@dataclass
class OneOrMore(Decorator):
    def format(self):
        return '(' + self.item.format() + ')+'


@dataclass
class ZeroOrMore(Decorator):
    def format(self):
        return '(' + self.item.format() + ')*'


@dataclass
class Gather(Node):
    separator: Node
    item: Node

    def __iter__(self):
        yield self.separator
        yield self.item

    def format(self):
        return f'({self.separator.format()}).({self.item.format()})+'


@dataclass
class Leaf(Node):
    """Node with no children"""
    value: str

    def format(self):
        return self.value

    def __iter__(self):
        return iter(())


@dataclass
class Reference(Leaf):
    def format(self):
        if self.value.startswith('invalid_'):
            raise InvalidRuleIncluded()
        return self.value


@dataclass
class String(Leaf):
    pass


def convert_grammar_node(grammar_node):
    """Convert a pegen grammar node to our AST node"""
    match grammar_node:
        case pegen.grammar.Rhs():
            return Choice([convert_grammar_node(alt) for alt in grammar_node.alts])
        case pegen.grammar.Alt():
            return Sequence([convert_grammar_node(item) for item in grammar_node.items])
        case pegen.grammar.Group():
            return convert_grammar_node(grammar_node.rhs)
        case pegen.grammar.NamedItem():
            return convert_grammar_node(grammar_node.item)
        case pegen.grammar.Opt():
            return Optional(convert_grammar_node(grammar_node.node))
        case pegen.grammar.NameLeaf():
            return Reference(grammar_node.value)
        case pegen.grammar.StringLeaf():
            return String(grammar_node.value)
        case pegen.grammar.Repeat1():
            return OneOrMore(convert_grammar_node(grammar_node.node))
        case pegen.grammar.Repeat0():
            return ZeroOrMore(convert_grammar_node(grammar_node.node))
        case pegen.grammar.PositiveLookahead():
            return Sequence([])
        case pegen.grammar.NegativeLookahead():
            return Sequence([])
        case pegen.grammar.Cut():
            return Sequence([])
        case pegen.grammar.Forced():
            return convert_grammar_node(grammar_node.node)
        case pegen.grammar.Gather():
            return Gather(
                convert_grammar_node(grammar_node.separator),
                convert_grammar_node(grammar_node.node),
            )
        case _:
            raise TypeError(f'{grammar_node!r} has unknown type {type(grammar_node).__name__}')


def generate_all_descendants(node):
    yield node
    for value in node:
        if isinstance(value, list):
            for item in value:
                yield from generate_all_descendants(item)
        else:
            yield from generate_all_descendants(value)

def generate_related_rules(rule, rules, top_level_rule_names, visited):
    if rule.name.startswith('invalid_'):
        return []
    if rule in visited:
        return []
    visited.add(rule)
    result = [
        (rule.name, convert_grammar_node(rule.rhs).format())
    ]
    for descendant in generate_all_descendants(rule):
        match descendant:
            case pegen.grammar.NameLeaf():
                if descendant.value in rules and descendant.value not in top_level_rule_names:
                    used_rule = rules[descendant.value]
                    print('matched', used_rule)
                    result.extend(generate_related_rules(used_rule, rules, top_level_rule_names, visited))
    return result


def generate_data(grammar, top_level_rule_names):
    result = {
        '#': 'This file was generated by docs_fragment_generator.py',
    }
    rules = dict(grammar.rules)
    for rule_name in top_level_rule_names:
        try:
            rule = rules[rule_name]
        except KeyError:
            raise KeyError(f'{rule_name} should appear in docs, but was not found in the grammar.')
        print(rule_name, rule)
        visited = set()
        result[rule_name] = generate_related_rules(rule, rules, top_level_rule_names, visited)

        print()
    return result

def main() -> None:
    args = argparser.parse_args()

    try:
        grammar, parser, tokenizer = build_parser(args.grammar_filename)
    except Exception as err:
        print("ERROR: Failed to parse grammar file", file=sys.stderr)
        sys.exit(1)

    with open(args.fragments_filename) as file:
        top_level_rule_names = []
        for line in file:
            # Remove comments (`#` and anything after it) & whitespace
            line = line.partition('#')[0].strip()
            if line:
                top_level_rule_names.append(line)

    result = generate_data(grammar, top_level_rule_names)
    with open(args.output_filename, 'w') as output_file:
        json.dump(result, output_file, indent=4)


if __name__ == "__main__":
    main()
