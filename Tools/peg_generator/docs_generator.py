from pathlib import Path
import re
import argparse
from dataclasses import dataclass
import enum
import collections

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
argparser.add_argument(
    '--debug', action='store_true',
    help="Include debug information in the generated docs.",
)


# TODO: Document all these rules somewhere in the docs
FUTURE_TOPLEVEL_RULES = {
    'statement', 'compound_stmt', 'simple_stmt', 'expression',
    't_primary', 'slices', 'star_expressions', 'with_item',
    'decorators', 'type_params', 'function_def',
    'if_stmt', 'class_def', 'with_stmt', 'for_stmt', 'try_stmt', 'while_stmt',
    'match_stmt', 'named_expression', 'star_targets', 'shift_expr',
    'bitwise_or', 'fstring_replacement_field', 'strings', 'literal_expr',
}

# TODO:
# simplify:
#   elif_stmt  ::=  'elif' named_expression ':' block (elif_stmt | [else_block])
# into:
#   elif_stmt  ::=  ('elif' named_expression ':' block)*  [else_block]
#
# don't simplify:
#   try_stmt          ::=  'try' ':' block (finally_block | (except_block+ ...
# instead keep 3 separate alternatives, like in the grammar
#
# Line-break with_stmt as:
# with_stmt ::=  ['async'] 'with'
#               ('(' ','.with_item+ [','] ')' | ','.with_item+)
#               ':' block
#
# for:
#   pattern_capture_target ::=  !'_' NAME
# we might want to show the negative lookahead

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
            ignoring = False
            for line in file:
                original_lines.append(line)
                if ignoring:
                    is_indented = (not line.strip()
                                   or line.startswith((' ', '\t')))
                    if not is_indented:
                        ignoring = False
                if not ignoring:
                    new_lines.append(line)
                if match := HEADER_RE.fullmatch(line):
                    ignoring = True
                    new_lines.append('   :group: python-grammar\n')
                    new_lines.append(f'   :generated-by: {SCRIPT_NAME}\n')
                    new_lines.append('\n')
                    for line in generate_rule_lines(
                        pegen_rules,
                        match[1].split(),
                        set(toplevel_rules) | FUTURE_TOPLEVEL_RULES,
                        debug=args.debug,
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


# TODO: Check parentheses are correct in complex cases.

class Precedence(enum.IntEnum):
    CHOICE = enum.auto()
    SEQUENCE = enum.auto()
    REPEAT = enum.auto()
    ATOM = enum.auto()

@dataclass(frozen=True)
class Node:
    def format_enclosed(self):
        return self.format()

    def simplify(self):
        return self

    def format_for_precedence(self, parent_precedence):
        result = self.format()
        if self.precedence < parent_precedence:
            result = '(' + result + ')'
        return result

    def dump_tree(self, indent=0):
        yield '  : ' + '  ' * indent + self.format()


@dataclass(frozen=True)
class Container(Node):
    """Collection of zero or more child items"""
    items: list[Node]

    def __iter__(self):
        yield from self.items

    def simplify_item(self, item):
        return item.simplify()

    def dump_tree(self, indent=0):
        yield '  : ' + '  ' * indent + type(self).__name__ + ':'
        for item in self:
            yield from item.dump_tree(indent + 1)

    def inlined(self, replaced_name, replacement):
        self_type = type(self)
        return self_type(
            [item.inlined(replaced_name, replacement) for item in self.items]
        )

@dataclass(frozen=True)
class Choice(Container):
    precedence = Precedence.CHOICE

    def format(self):
        return " | ".join(
            item.format_for_precedence(Precedence.CHOICE)
            for item in self
        )

    def simplify(self):
        self_type = type(self)
        alternatives = []
        is_optional = False
        for item in self:
            item = self.simplify_item(item)
            match item:
                case None:
                    pass
                case Container([]):
                    is_optional = True
                    # ignore the item
                case Sequence(elements):
                    alternatives.append(elements)
                case _:
                    alternatives.append([item])
        assert all(isinstance(item, list) for item in alternatives)


        # Simplify subsequences: call simplify_subsequence on all
        # "tails" of `alternatives`.
        new_alts = []
        index = 0
        while index < len(alternatives):
            replacement, num_processed = self.simplify_subsequence(alternatives[index:])

            # replacement should be list[list[Node]]
            assert isinstance(replacement, list)
            assert all(isinstance(alt, list) for alt in replacement)
            assert all(isinstance(node, Node) for alt in replacement for node in alt)

            # Ensure we make progress
            assert num_processed > 0

            new_alts.extend(replacement)
            index += num_processed
        alternatives = new_alts


        def wrap(node):
            if is_optional:
                return Optional(node)
            else:
                return node

        if len(alternatives) == 1:
            return wrap(Sequence(alternatives[0]))
        if not alternatives:
            return Sequence([])
        first_alt = alternatives[0]
        if all(alt[0] == first_alt[0] for alt in alternatives[1:]):
            return wrap(Sequence([
                first_alt[0],
                Choice([
                    Sequence(alt[1:]) for alt in alternatives
                ]),
            ]))
        if all(alt[-1] == first_alt[-1] for alt in alternatives[1:]):
            return wrap(Sequence([
                Choice([
                    Sequence(alt[:-1]) for alt in alternatives
                ]),
                first_alt[-1],
            ]))

        return wrap(self_type(
            [Sequence(alt).simplify() for alt in alternatives]
        ))

    def simplify_item(self, item):
        match item:
            case Sequence([Nonterminal(name)]) if name.startswith('invalid_'):
                return None
        return super().simplify_item(item)

    def simplify_subsequence(self, tail):

        match tail[:2]:
            case [
                [x],
                [Gather(_, x1), Optional()] as result,
            ] if x == x1:
                return [result], 2
            case [
                [*h1, common_last_node1],
                [*h2, common_last_node2],
            ] if common_last_node1 == common_last_node2:
                result = [[Choice([Sequence(h1), Sequence(h2)]), common_last_node1]], 2
                return result

        return [tail[0]], 1

@dataclass(frozen=True)
class Sequence(Container):
    precedence = Precedence.SEQUENCE

    def simplify(self):
        self_type = type(self)
        items = []
        for item in self:
            item = self.simplify_item(item)
            match item:
                case Container([]):
                    pass
                case Sequence(subitems):
                    items.extend(self.simplify_item(si) for si in subitems)
                case _:
                    items.append(item)
        if not items:
            return Sequence([])

        # Simplify subsequences: call simplify_subsequence on all
        # "tails" of `items`.
        new_items = []
        index = 0
        while index < len(items):
            replacement, num_processed = self.simplify_subsequence(items[index:])

            # Single nodes are iterable; they act as a sequence of their
            # children, and we don't want that in this case.
            assert not isinstance(replacement, Node)

            # Ensure we make progress
            assert num_processed > 0

            new_items.extend(replacement)
            index += num_processed
        items = new_items

        if len(items) == 1:
            return items[0]
        return self_type(items)

    def simplify_subsequence(self, subsequence):
        """Simplify the start of the given (sub)sequence.

        Return the simplified result and the number of items that were
        simplified.
        """
        match subsequence[:2]:
            case [e1, ZeroOrMore(Sequence([s, e2]))] if e1 == e2:
                return [Gather(s, e2)], 2
        return [subsequence[0]], 1

    def format(self):
        return " ".join(
            item.format_for_precedence(Precedence.SEQUENCE)
            for item in self
        )

@dataclass(frozen=True)
class Decorator(Node):
    """Node with exactly one child"""
    item: Node

    def __iter__(self):
        yield self.item

    def dump_tree(self, indent=0):
        yield '  : ' + '  ' * indent + type(self).__name__ + ':'
        yield from self.item.dump_tree(indent + 1)

    def simplify(self):
        self_type = type(self)
        item = self.item.simplify()
        match item:
            case Sequence([x]):
                item = x
            case Sequence([]):
                return Sequence([])
        return self_type(item)

    def inlined(self, replaced_name, replacement):
        self_type = type(self)
        return self_type(self.item.inlined(replaced_name, replacement))

@dataclass(frozen=True)
class Optional(Decorator):
    precedence = Precedence.ATOM

    def format(self):
        return '[' + self.item.format() + ']'

    def simplify(self):
        match self.item:
            #  [x [y] | y]  ->  [x] [y]
            case Choice([Sequence([x, Optional(y1)]), y2]) if y1 == y2:
                return Sequence([Optional(x), Optional(y1)])
            case OneOrMore(x):
                return ZeroOrMore(x)
        return super().simplify()


@dataclass(frozen=True)
class OneOrMore(Decorator):
    precedence = Precedence.REPEAT

    def format(self):
        return self.item.format_for_precedence(Precedence.REPEAT) + '+'


@dataclass(frozen=True)
class ZeroOrMore(Decorator):
    precedence = Precedence.REPEAT

    def format(self):
        return self.item.format_for_precedence(Precedence.REPEAT) + '*'

@dataclass(frozen=True)
class Gather(Node):
    separator: Node
    item: Node

    precedence = Precedence.REPEAT

    def __iter__(self):
        yield self.separator
        yield self.item

    def simplify(self):
        self_type = type(self)
        return self_type(self.separator.simplify(), self.item.simplify())

    def format(self):
        sep_rep = self.separator.format_for_precedence(Precedence.REPEAT)
        item_rep = self.item.format_for_precedence(Precedence.REPEAT)
        return f'{sep_rep}.{item_rep}+'

    def dump_tree(self, indent=0):
        yield '  : ' + '  ' * indent + type(self).__name__ + ':'
        yield from self.item.dump_tree(indent + 1)
        yield '  : ' + '  ' * indent + 'separator:'
        yield from self.separator.dump_tree(indent + 1)

    def inlined(self, replaced_name, replacement):
        self_type = type(self)
        return self_type(
            self.separator.inlined(replaced_name, replacement),
            self.item.inlined(replaced_name, replacement),
        )

@dataclass(frozen=True)
class Leaf(Node):
    """Node with no children"""
    value: str

    precedence = Precedence.ATOM

    def format(self):
        return self.value

    def __iter__(self):
        return iter(())

    def inlined(self, replaced_name, replacement):
        return self


@dataclass(frozen=True)
class Token(Leaf):
    pass


@dataclass(frozen=True)
class Nonterminal(Leaf):
    def format(self):
        return f"`{self.value}`"

    def inlined(self, replaced_name, replacement):
        if self.value == replaced_name:
            return replacement
        return super().inlined(replaced_name, replacement)


@dataclass(frozen=True)
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

def generate_rule_lines(pegen_rules, rule_names, toplevel_rule_names, debug):
    # Figure out all rules we want to document.
    # This includes the ones we were asked to document (`requested_rule_names`),
    # and also rules referenced from them (recursively), except ones that will
    # be documented elsewhere (those in `toplevel_rule_names`).
    requested_rule_names = list(rule_names)
    rule_names_to_add = list(rule_names)
    ruleset = {}
    while rule_names_to_add:
        rule_name = rule_names_to_add.pop(0)
        if rule_name in ruleset:
            continue
        pegen_rule = pegen_rules[rule_name]
        node = convert_grammar_node(pegen_rule.rhs)

        node = simplify_node(node)
        ruleset[rule_name] = node

        for descendant in generate_all_descendants(node, filter=Nonterminal):
            rule_name = descendant.value
            if (
                rule_name in pegen_rules
                and rule_name not in toplevel_rule_names
            ):
                rule_names_to_add.append(rule_name)

    # Simplify ruleset repeatedly, until simplification no longer changes it
    old_ruleset = None
    while ruleset != old_ruleset:
        old_ruleset = ruleset
        ruleset = simplify_ruleset(ruleset, requested_rule_names)

    # Yield all the lines
    for name, node in ruleset.items():
        if debug:
            # To compare with pegen's stringification:
            yield f'{name} (from pegen): {pegen_rules[name]!r}'
            yield f'{name} (repr): {node!r}'

        rhs_line = node.format()
        if isinstance(node, Choice) and len(rhs_line) > 40:
            # Present each alternative on its own line
            yield f'{name}:'
            for alt in node:
                yield f'  : | {alt.format()}'
        else:
            yield f'{name}: {node.format()}'
        if debug:
            yield from node.dump_tree()

def simplify_node(node):
    """Simplifies a node repeatedly until simplification no longer changes it"""
    last_node = None
    while node != last_node:
        last_node = node
        node = node.simplify()
    return node


def simplify_ruleset(old_ruleset: dict, requested_rule_names):
    """Simplify and inline a bunch of rules"""

    # Generate new ruleset with simplified nodes
    new_ruleset = {}
    for rule_name, node in old_ruleset.items():
        new_ruleset[rule_name] = simplify_node(node)

    # Count up all the references to rules we're documenting.
    # A rule that's only mentioned once should be inlined, except if we were
    # explicitly asked to provide a definition for it (i.e. it is in
    # `requested_rule_names`).
    reference_counts = collections.Counter()
    for node in new_ruleset.values():
        for descendant in generate_all_descendants(node, filter=Nonterminal):
            name = descendant.value
            if name in new_ruleset and name not in requested_rule_names:
                reference_counts[name] += 1

    # Inline the rules we found
    for name, count in reference_counts.items():
        if count == 1:
            print(f'rule named {name} will be inlined.')
            replaced_name = name
            replacement = new_ruleset[replaced_name]
            for rule_name, rule_node in new_ruleset.items():
                new_node = rule_node.inlined(replaced_name, replacement)
                new_ruleset[rule_name] = new_node
            del new_ruleset[name]

    return new_ruleset

def generate_all_descendants(node, filter=Node):
    if isinstance(node, filter):
        yield node
    for value in node:
        yield from generate_all_descendants(value, filter)

if __name__ == "__main__":
    main()
