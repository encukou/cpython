import argparse
import sys
from typing import Any, Callable, Iterator
import json

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


def format_node(node):
    match node:
        case pegen.grammar.Rhs():
            option_reprs = []
            for alt in node.alts:
                try:
                    option_reprs.append(format_node(alt))
                except InvalidRuleIncluded:
                    pass
            return " | ".join(option_reprs)
        case pegen.grammar.Alt():
            item_reprs = []
            for item in node.items:
                item_repr = format_node(item)
                if item_repr:
                    item_reprs.append(item_repr)
            return " ".join(item_reprs)
        case pegen.grammar.NamedItem():
            return format_node(node.item)
        case pegen.grammar.Opt():
            return '[' + format_node(node.node) + ']'
        case pegen.grammar.NameLeaf():
            if node.value.startswith('invalid_'):
                raise InvalidRuleIncluded
            return node.value
        case pegen.grammar.StringLeaf():
            return node.value
        case pegen.grammar.Repeat1():
            return format_node(node.node) + '+'
        case pegen.grammar.Repeat0():
            return format_node(node.node) + '*'
        case pegen.grammar.Group():
            return '(' + format_node(node.rhs) + ')'
        case pegen.grammar.Gather():
            return f"{node.separator!s}.{node.node!s}+"
        case pegen.grammar.Forced():
            # Ignore forced-ness of a node
            return format_node(node.node)
        case pegen.grammar.PositiveLookahead() | pegen.grammar.NegativeLookahead() | pegen.grammar.Cut():
            # Ignore advanced features for now
            return ''
        case _:
            raise TypeError(f'{node!r} has unknown type {type(node).__name__}')


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
        (rule.name, format_node(rule.rhs))
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
