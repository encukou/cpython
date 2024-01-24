from pathlib import Path
import re
import argparse
from dataclasses import dataclass

import pegen.grammar
from pegen.build import build_parser

# TODO: handle indentation
HEADER_RE = re.compile(r'..\s+grammar-snippet\s*::(.*)', re.DOTALL)

# Hardcoded so it's the same regardless of how this script is invoked
SCRIPT_NAME = 'Tools/peg_generator/docs_generator.py'

argparser = argparse.ArgumentParser(
    prog="docz_generator.py",
    description="Re-generate the grammar snippets in docs",
)
argparser.add_argument("grammar_filename", help="Grammar description")
argparser.add_argument(
    "docs_dir",
    help="Directory with the docs. All .rst files in this "
        + "(and subdirs) will be regenerated.",
)


# TODO: Document all these rules somewhere in the docs
FUTURE_TOPLEVEL_RULES = {
    'statement', 'compound_stmt', 'simple_stmt', 'expression',
}


def main():
    args = argparser.parse_args()

    # Get all the top-level rule names, and the files we're updating

    files_with_grammar = set()
    # Maps the name of a top-level rule to the path of the file it's in
    toplevel_rules = {}
    for path in Path(args.docs_dir).glob('**/*.rst'):
        with path.open(encoding='utf-8') as file:
            for line in file:
                if match := HEADER_RE.fullmatch(line):
                    files_with_grammar.add(path)
                    for rule in match[1].split():
                        if rule in toplevel_rules:
                            raise ValueError(
                                f'rule {rule!r} appears both in '
                                + f'{toplevel_rules[rule]} and in {path}. It '
                                + f'should only be documented in one place.')
                        toplevel_rules[rule] = path
    print(f'{toplevel_rules=}')

    # Parse the grammar

    grammar, parser, tokenizer = build_parser(args.grammar_filename)
    pegen_rules = dict(grammar.rules)

    # Update the files

    for path in files_with_grammar:
        with path.open(encoding='utf-8') as file:
            original_lines = []
            new_lines = []
            blocks_to_ignore = 0
            for line in file:
                original_lines.append(line)
                if blocks_to_ignore:
                    if not line.strip() and blocks_to_ignore > 0:
                        blocks_to_ignore -= 1
                else:
                    new_lines.append(line)
                if match := HEADER_RE.fullmatch(line):
                    blocks_to_ignore = 2
                    new_lines.append('   :group: python-grammar\n')
                    new_lines.append(f'   :generated-by: {SCRIPT_NAME}\n')
                    new_lines.append('\n')
                    for line in generate_rule_lines(
                        pegen_rules,
                        match[1].split(),
                        set(toplevel_rules) | FUTURE_TOPLEVEL_RULES,
                    ):
                        new_lines.append(f'   {line}\n')
                    new_lines.append('\n')

        # TODO: It would be nice to trim final blank lines,
        # but that causes the final line to be read twice by Sphinx,
        # adding it as a blockquote. This might be an issue in the extension.
        #while new_lines and not new_lines[-1].strip():
        #    del new_lines[-1]

        if original_lines != new_lines:
            print(f'Updating: {path}')
            with path.open(encoding='utf-8', mode='w') as file:
                file.writelines(new_lines)
        else:
            print(f'Unchanged: {path}')


@dataclass
class Node:
    def format_enclosed(self):
        return self.format()

    def simplify(self):
        return self

@dataclass
class Container(Node):
    """Collection of zero or more child items"""
    items: list[Node]

    def __iter__(self):
        yield from self.items

    def format_enclosed(self):
        if len(self.items) > 1:
            return '(' + self.format() + ')'
        return self.format()

    def simplify(self):
        self_type = type(self)
        items = []
        for item in self:
            item = item.simplify()
            match item:
                case Container([]):
                    pass
                case _:
                    items.append(item)
        return self_type(items)

@dataclass
class Choice(Container):
    def format(self):
        return " | ".join([item.format_enclosed() for item in self])

@dataclass
class Sequence(Container):
    def format(self):
        # TODO: Sometimes add parentheses
        return " ".join([item.format_enclosed() for item in self])

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
        if isinstance(self.item, Gather):
            # TODO:
            raise AssertionError('Gather in OneOrMore, check how parentheses should be added')
        return self.item.format_enclosed() + '+'


@dataclass
class ZeroOrMore(Decorator):
    def format(self):
        if isinstance(self.item, Gather):
            # TODO:
            raise AssertionError('Gather in ZeroOrMore, check how parentheses should be added')
        return self.item.format_enclosed() + '*'

@dataclass
class Gather(Node):
    separator: Node
    item: Node

    def __iter__(self):
        yield self.separator
        yield self.item

    def format(self):
        sep = self.separator.format_enclosed()
        item = self.item.format_enclosed()
        return f'{sep}.{item}+'

@dataclass
class Leaf(Node):
    """Node with no children"""
    value: str

    def format(self):
        return self.value

    def __iter__(self):
        return iter(())


@dataclass
class Token(Leaf):
    pass


@dataclass
class Nonterminal(Leaf):
    def format(self):
        return f"`{self.value}`"


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
            if grammar_node.value == 'TYPE_COMMENT':
                return Sequence([])
            if grammar_node.value.isupper():
                return Token(grammar_node.value)
            else:
                return Nonterminal(grammar_node.value)
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

def generate_rule_lines(pegen_rules, rule_names, toplevel_rule_names):
    rule_names = list(rule_names)
    seen_rule_names = set()
    while rule_names:
        rule_name = rule_names.pop(0)
        if rule_name in seen_rule_names:
            continue
        pegen_rule = pegen_rules[rule_name]
        node = convert_grammar_node(pegen_rule.rhs).simplify()
        print(pegen_rule.name)
        print(node)
        yield f'{pegen_rule.name}: {node.format()}'
        seen_rule_names.add(rule_name)

        for descendant in generate_all_descendants(node):
            if isinstance(descendant, Nonterminal):
                rule_name = descendant.value
                if (
                    rule_name in pegen_rules
                    and rule_name not in toplevel_rule_names
                ):
                    rule_names.append(rule_name)

def generate_all_descendants(node):
    yield node
    for value in node:
        yield from generate_all_descendants(value)

if __name__ == "__main__":
    main()
