from pathlib import Path
import re
import argparse
from dataclasses import dataclass
import enum
import collections
import typing
import tokenize
import token

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
FUTURE_TOPLEVEL_RULES = set()


# Rules that should be replaced by another rule, *if* they have the
# same contents.
# (In the grammar, these rules may have different actions, which we ignore
# here.)
REPLACED_SYNONYMS = {
    'literal_expr': 'literal_pattern',
}

# TODO:

# Think about "OptionalSequence with separators":

# Gather:
# s.e+
#   Match one or more occurrences of e, separated by s. The generated parse tree
#   does not include the separator. This is otherwise identical to (e (s e)*).
#
# Proposal: Optional sequence; but for diagrams only:
# s.{e1, e2, e3}
#     Match the given expressions in the given order, separated by separator.
#     Each of the expressions is individually optional, but at least one must be given.
#     Equavalent to:
#       e1 [s e2 [s e3]]
#     |       e2 [s e3]
#     |             e3
#     (There can be any nonzero number of e_n, not necessarily 3)
#
#
#      [[e1 s] e2 s] e3
#     | [e1 s] e2 s
#     |  e1


# TODO:
# Better line wrapping
#
# Line-break with_stmt as:
# with_stmt ::=  ['async'] 'with'
#               ('(' ','.with_item+ [','] ')' | ','.with_item+)
#               ':' block
#
# simplify:
#   elif_stmt  ::=  'elif' named_expression ':' block [elif_stmt | else_block]
# into:
#   elif_stmt  ::=  ('elif' named_expression ':' block)+  [else_block]
#
# Look at function parameters again
#

# NEED GRAMMAR CHANGES:
#
# (leave for later)
# Mark an alternative as optimization only, so it doesn't show up in docs.
# For example, in del_t_atom, the rule
#   '(' del_target ')'
# is covered by the next rule:
#    '(' [del_targets] ')'
#
# (leave for later)
# Give names to the subexpressions here:
# proper_slice ::=  [lower_bound] ":" [upper_bound] [ ":" [stride] ]
#
# (leave for later)
# don't simplify:
#   try_stmt          ::=  'try' ':' block (finally_block | (except_block+ ...
# instead keep 3 separate alternatives, like in the grammar
# Similar for star_targets_tuple_seq


def main():
    args = argparser.parse_args()

    # Get all the top-level rule names, and the files we're updating
    # "top-level" means a rule that the docs explicitly ask for in
    # a `grammar-snippet` directive. Anything that references a top-level
    # rule should link to it.

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

    rules = convert_grammar(pegen_rules, toplevel_rules)

    if args.debug:
        for name, rule in rules.items():
            print()
            print(name, rule.format())
            for line in rule.dump_tree():
                print(line)

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
                        rules,
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
    LOOKAHEAD = enum.auto()
    ATOM = enum.auto()

@dataclass(frozen=True)
class Node:
    def format_enclosed(self):
        return self.format()

    def simplify(self, rules, path):
        return self

    def format_for_precedence(self, parent_precedence):
        result = self.format()
        if self.precedence < parent_precedence:
            result = '(' + result + ')'
        return result

    def dump_tree(self, indent=0):
        yield '  : ' + '  ' * indent + self.format()

    def format_lines(self, columns):
        yield self.format()


@dataclass(frozen=True)
class Container(Node):
    """Collection of zero or more child items"""
    items: list[Node]

    def __iter__(self):
        yield from self.items

    def simplify_item(self, rules, item, path):
        return simplify_node(rules, item, path)

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
        if not self.items:
            return '<UNREACHABLE>'
        return " | ".join(
            item.format_for_precedence(Precedence.CHOICE)
            for item in self
        )

    def simplify(self, rules, path):
        self_type = type(self)
        alternatives = []
        is_optional = False
        for i, item in enumerate(self):
            item = self.simplify_item(rules, item, path.child(self, i))
            match item:
                case None:
                    pass
                case self.EMPTY:
                    is_optional = True
                    # ignore the item
                case self.UNREACHABLE:
                    # ignore the item
                    pass
                case Optional(x):
                    is_optional = True
                    alternatives.append([x])
                case Sequence(elements):
                    alternatives.append(elements)
                case Choice(sub_alts):
                    alternatives.extend([a] for a in sub_alts)
                case _:
                    alternatives.append([item])
        assert all(isinstance(item, list) for item in alternatives)

        # Simplify subsequences: call simplify_subsequence on all
        # "tails" of `alternatives`.
        new_alts = []
        index = 0
        while index < len(alternatives):
            replacement, num_processed = self.simplify_subsequence(rules, alternatives[index:])

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

        return wrap(self_type([
            simplify_node(rules, Sequence(alt), path.child(self, None))
            for alt in alternatives
        ]))

    def simplify_subsequence(self, rules, tail):
        if len(tail) >= 2:
            # If two or more adjacent alternatives start or end with the
            # same item, we pull that item out, and replace the alternatives
            # with a sequence of
            # [common item, Choice of the remainders of the alts].
            first_alt = tail[0]
            # We do this for both the start and the end; for that we need the
            # index of the candidate item (0 or -1) and the slice to get the
            # rest of the items.
            for index, rest_slice in (
                    (0, slice(1, None)),
                    (-1, slice(None, -1)),
                ):
                num_alts_with_common_item = 1
                for alt in tail[1:]:
                    if alt[index] != first_alt[index]:
                        break
                    num_alts_with_common_item += 1
                if num_alts_with_common_item > 1:
                    common_item = first_alt[index]
                    remaining_choice = Choice([
                        Sequence(alt[rest_slice])
                        for alt in tail[:num_alts_with_common_item]
                    ])
                    if index == 0:
                        result = [
                            [common_item, remaining_choice],
                        ]
                    else:
                        result = [
                            [remaining_choice, common_item],
                        ]
                    return result, num_alts_with_common_item

        match tail[:2]:
            case [
                [x],
                [Gather(_, x1), Optional()] as result,
            ] if x == x1:
                return [result], 2

        return [tail[0]], 1

    def _format_lines(self, columns):
        simple = self.format()
        if len(simple) <= columns:
            yield simple
        else:
            yield ''
            for choice in self:
                for num, line in enumerate(choice.format_lines(columns - 2)):
                    if num == 0:
                        yield '| ' + line
                    else:
                        yield '  ' + line

    def get_possible_start_tokens(self, rules, rules_considered):
        result = set()
        for item in self.items:
            result.update(item.get_possible_start_tokens(rules, rules_considered))
        return result

    def get_follow_set_for_path(self, path, rules):
        return get_follow_set_for_path(path.parent_entry, rules)

@dataclass(frozen=True)
class Sequence(Container):
    precedence = Precedence.SEQUENCE

    def simplify(self, rules, path):
        self_type = type(self)
        items = []
        for i, item in enumerate(self):
            item = self.simplify_item(rules, item, path.child(self, i))
            match item:
                case self.EMPTY:
                    pass
                case self.UNREACHABLE:
                    return UNREACHABLE
                case Sequence(subitems):
                    items.extend(subitems)
                case _:
                    items.append(item)
        if not items:
            return EMPTY

        # Simplify subsequences: call simplify_subsequence on all
        # "tails" of `items`.
        new_items = []
        index = 0
        while index < len(items):
            replacement, num_processed = self.simplify_subsequence(rules, items[index:])

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

    def simplify_subsequence(self, rules, subsequence):
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

    def _format_lines(self, columns):
        simple = self.format()
        if len(simple) <= columns:
            yield simple
        else:
            line_so_far = ''
            for item in self:
                simple_item = item.format()
                if len(line_so_far) + len(simple_item) < columns:
                    line_so_far += ' ' + simple_item
                else:
                    yield line_so_far
                    line_so_far = ''
                    if len(simple_item) > columns:
                        for line in item.format_lines(columns):
                            yield line
                    else:
                        line_so_far = simple_item
            if line_so_far:
                yield line_so_far

    def get_possible_start_tokens(self, rules, rules_considered):
        if not self.items:
            return {None}
        result = set()
        for item in self.items:
            item_start_tokens = item.get_possible_start_tokens(rules, rules_considered)
            result.update(item_start_tokens)
            item_can_be_empty = (None in item_start_tokens)
            if item_can_be_empty:
                continue
            else:
                result.discard(None)
                break
        return result

    def get_follow_set_for_path(self, path, rules):
        assert path.node == self
        items = self.items[path.position+1:]
        result = {None}
        for item in items:
            item_start_tokens = item.get_possible_start_tokens(rules, set())
            result.update(item_start_tokens)
            item_can_be_empty = (None in item_start_tokens)
            if item_can_be_empty:
                continue
            else:
                result.discard(None)
                break
        if None in result:
            result.discard(None)
            result.update(get_follow_set_for_path(path.parent_entry, rules))
        return result


@dataclass(frozen=True)
class Decorator(Node):
    """Node with exactly one child"""
    item: Node

    def __iter__(self):
        yield self.item

    def dump_tree(self, indent=0):
        yield '  : ' + '  ' * indent + type(self).__name__ + ':'
        yield from self.item.dump_tree(indent + 1)

    def simplify(self, rules, path):
        self_type = type(self)
        item = simplify_node(rules, self.item, path.child(self))
        match item:
            case Sequence([x]):
                item = x
            case self.EMPTY:
                return EMPTY
        return self_type(item)

    def inlined(self, replaced_name, replacement):
        self_type = type(self)
        return self_type(self.item.inlined(replaced_name, replacement))

@dataclass(frozen=True)
class Optional(Decorator):
    precedence = Precedence.ATOM

    def format(self):
        return '[' + self.item.format() + ']'

    def simplify(self, rules, path):
        match self.item:
            #  [x [y] | y]  ->  [x] [y]
            case Choice([Sequence([x, Optional(y1)]), y2]) if y1 == y2:
                return Sequence([Optional(x), Optional(y1)])
            case OneOrMore(x):
                return ZeroOrMore(x)
            case Optional(x):
                return self.item
            case self.UNREACHABLE:
                return EMPTY
        return super().simplify(rules, path)

    def get_possible_start_tokens(self, rules, rules_considered):
        return self.item.get_possible_start_tokens(rules, rules_considered) | {None}


@dataclass(frozen=True)
class OneOrMore(Decorator):
    precedence = Precedence.REPEAT

    def format(self):
        return self.item.format_for_precedence(Precedence.REPEAT) + '+'

    def get_possible_start_tokens(self, rules, rules_considered):
        return self.item.get_possible_start_tokens(rules, rules_considered)

    def get_follow_set_for_path(self, path, rules):
        item_start_tokens = self.item.get_possible_start_tokens(rules, set())
        item_start_tokens.discard(None)
        return (
            item_start_tokens
            | get_follow_set_for_path(path.parent_entry, rules)
        )

@dataclass(frozen=True)
class ZeroOrMore(Decorator):
    precedence = Precedence.REPEAT

    def format(self):
        return self.item.format_for_precedence(Precedence.REPEAT) + '*'

    def get_possible_start_tokens(self, rules, rules_considered):
        return self.item.get_possible_start_tokens(rules, rules_considered) | {None}

    def get_follow_set_for_path(self, path, rules):
        item_start_tokens = self.item.get_possible_start_tokens(rules, set())
        item_start_tokens.discard(None)
        return (
            item_start_tokens
            | get_follow_set_for_path(path.parent_entry, rules)
        )


@dataclass(frozen=True)
class Lookahead(Decorator):
    precedence = Precedence.LOOKAHEAD

    def format(self):
        return self.sigil + self.item.format_for_precedence(Precedence.LOOKAHEAD)

    def get_possible_start_tokens(self, rules, rules_considered):
        return {None}

    def tokens_in_self(self, rules):
        """Return the set of tokens contained in this lookahead, or None if
        the lookahead can match more than one token"""
        item = self.item
        if isinstance(item, Nonterminal):
            item = rules[item.value]
        if isinstance(item, Choice):
            tokens_in_self = item.items
        else:
            tokens_in_self = {item}
        if not all(isinstance(item, BaseToken) for item in tokens_in_self):
            # we only simplify lookaheads that only contain tokens
            return None
        return TokenSet(tokens_in_self)


class NegativeLookahead(Lookahead):
    sigil = '!'

    def simplify(self, rules, path):
        # Find all the tokens the lookahead contains
        # (We don't simplify lookaheads that contain more than tokens)
        tokens_in_self = self.tokens_in_self(rules)
        if tokens_in_self:
            self_follow_set = TokenSet(get_follow_set_for_path(path, rules))

            if not tokens_in_self.issubset(self_follow_set):
                # no tokens in the neg.lookahead can follow it,
                # so the lookahead is redundant
                return EMPTY
        return super().simplify(rules, path)

class PositiveLookahead(Lookahead):
    sigil = '&'

    def simplify(self, rules, path):
        # Find all the tokens the lookahead contains
        # (We don't simplify lookaheads that contain more than tokens)
        tokens_in_self = self.tokens_in_self(rules)
        if tokens_in_self:
            self_follow_set = TokenSet(get_follow_set_for_path(path, rules))

            if tokens_in_self.issuperset(self_follow_set):
                # no other tokens than what's in the lookahead can follow it,
                # so the lookahead is redundant
                return EMPTY
        return super().simplify(rules, path)


@dataclass(frozen=True)
class Gather(Node):
    separator: Node
    item: Node

    precedence = Precedence.REPEAT

    def __iter__(self):
        yield self.separator
        yield self.item

    def simplify(self, rules, path):
        self_type = type(self)
        return self_type(
            simplify_node(rules, self.separator, path.child(self, 'sep')),
            simplify_node(rules, self.item, path.child(self, 'item')),
        )

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

    def get_possible_start_tokens(self, rules, rules_considered):
        result = self.item.get_possible_start_tokens(rules, rules_considered)
        if None in result:
            result.remove(None)
            result.update(self.separator.get_possible_start_tokens(rules, rules_considered))
        return result

    def get_follow_set_for_path(self, path, rules):
        sep_start_tokens = self.separator.get_possible_start_tokens(rules, set())
        item_start_tokens = self.item.get_possible_start_tokens(rules, set())
        match path.position:
            case 'sep':
                result = set(item_start_tokens)
                if None in result:
                    result.update(sep_start_tokens)
                    result.update(
                        get_follow_set_for_path(path.parent_entry, rules)
                    )
                    result.discard(None)
                return result
            case 'item':
                result = set(sep_start_tokens)
                if None in result:
                    result.update(item_start_tokens)
                    result.discard(None)
                result.update(get_follow_set_for_path(path.parent_entry, rules))
                return result
            case _:
                raise ValueError(path.position)

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
class BaseToken(Leaf):
    def get_possible_start_tokens(self, rules, rules_considered):
        return {self}

class LiteralToken(BaseToken):
    """A token that matches an exact string, like "_" or ','or 'if'
    """
    @property
    def kind(self):
        # Remove quotes
        assert self.value[0] in ("'", '"')
        assert self.value[0] == self.value[-1]
        content = self.value[1:-1]
        kinds = {
            '[': "OP",  ']': "OP",
            '(': "OP",  ')': "OP",
            '{': "OP",  '}': "OP",
        }
        result = kinds.get(content)
        if result is not None:
            return result
        # Tokenize the result. The tokenize module is meant to be used on
        # files, not single tokens, so this is cumbersome.
        # We get extra NEWLINE and ENDMARKER tokens after the one we want.
        try:
            tok, newline, end = tokenize.generate_tokens(iter([content]).__next__)
        except:
            raise NotImplementedError(
                f'kind of token {content!r} could not be determined. '
                + 'special case it!')
        return token.tok_name[tok.type]

class SymbolicToken(BaseToken):
    """A token name, like NAME or NUMBER or NEWLINE
    """


@dataclass(frozen=True)
class Nonterminal(Leaf):
    def format(self):
        return f"`{self.value}`"

    def inlined(self, replaced_name, replacement):
        if self.value == replaced_name:
            return replacement
        return super().inlined(replaced_name, replacement)

    def get_possible_start_tokens(self, rules, rules_considered):
        if self.value in rules_considered:
            # don't recurse into a rule we're already evaluating
            return set()
        rule = rules[self.value]
        return rule.get_possible_start_tokens(rules, rules_considered | {self.value})

    def simplify(self, rules, path):
        if other_rule_name := REPLACED_SYNONYMS.get(self.value):
            self_rule = rules[self.value]
            other_rule = rules[other_rule_name]
            if self_rule == other_rule:
                return Nonterminal(other_rule_name)
        return super().simplify(rules, path)

EMPTY = Node.EMPTY = Sequence([])
UNREACHABLE = Node.UNREACHABLE = Choice([])

def convert_grammar_node(grammar_node):
    """Convert a pegen grammar node to our AST node"""
    match grammar_node:
        case pegen.grammar.Rhs():
            return Choice([convert_grammar_node(alt) for alt in grammar_node.alts])
        case pegen.grammar.Alt():
            if 'RAISE_SYNTAX_ERROR' in (grammar_node.action or ''):
                # This is actually invalid syntax,
                # see https://github.com/python/cpython/issues/118235
                return UNREACHABLE
            return Sequence([convert_grammar_node(item) for item in grammar_node.items])
        case pegen.grammar.Group():
            return convert_grammar_node(grammar_node.rhs)
        case pegen.grammar.NamedItem():
            return convert_grammar_node(grammar_node.item)
        case pegen.grammar.Opt():
            return Optional(convert_grammar_node(grammar_node.node))
        case pegen.grammar.NameLeaf():
            if grammar_node.value == 'TYPE_COMMENT':
                # The tokenizer doesn't emit TYPE_COMMENT unless it's in
                # a special mode
                return UNREACHABLE
            if grammar_node.value.isupper():
                return SymbolicToken(grammar_node.value)
            if grammar_node.value.startswith('invalid'):
                return UNREACHABLE
            else:
                return Nonterminal(grammar_node.value)
        case pegen.grammar.StringLeaf():
            return LiteralToken(grammar_node.value)
        case pegen.grammar.Repeat1():
            return OneOrMore(convert_grammar_node(grammar_node.node))
        case pegen.grammar.Repeat0():
            return ZeroOrMore(convert_grammar_node(grammar_node.node))
        case pegen.grammar.PositiveLookahead():
            return PositiveLookahead(convert_grammar_node(grammar_node.node))
        case pegen.grammar.NegativeLookahead():
            return NegativeLookahead(convert_grammar_node(grammar_node.node))
        case pegen.grammar.Cut():
            return EMPTY
        case pegen.grammar.Forced():
            return convert_grammar_node(grammar_node.node)
        case pegen.grammar.Gather():
            return Gather(
                convert_grammar_node(grammar_node.separator),
                convert_grammar_node(grammar_node.node),
            )
        case _:
            raise TypeError(f'{grammar_node!r} has unknown type {type(grammar_node).__name__}')

@dataclass(frozen=True)
class PathEntry:
    parent_entry: typing.Union["PathEntry", None]
    node: Node | None  # (None stands for the root -- the entire grammar)
    position: object

    def child(self, node, position=None):
        return PathEntry(self, node, position)

    @classmethod
    def root_path(cls, rule_name):
        return cls(None, None, rule_name)


class TokenSet:
    """A set of tokens, whose operations take into account that, for example,
    a NAME token is actually a set that includes e.g. "_"
    """
    def __init__(self, tokens):
        self.tokens = set(tokens)

    def issubset(self, other):
        # each element of self needs to appear in other
        for element in self.tokens:
            if element in other.tokens:
                continue
            if isinstance(element, LiteralToken):
                if SymbolicToken(element.kind) not in other.tokens:
                    return False
            else:
                return False
        return True

    def issuperset(self, other):
        return other.issubset(self)

def convert_grammar(pegen_rules, toplevel_rule_names):
    ruleset = {}
    for rule_name, pegen_rule in pegen_rules.items():
        node = convert_grammar_node(pegen_rule.rhs)
        ruleset[rule_name] = node

    for name, node in ruleset.items():
        path = PathEntry.root_path(name)
        ruleset[name] = simplify_node(ruleset, node, path)

    # Simplify ruleset repeatedly, until simplification no longer changes it
    old_ruleset = None
    while ruleset != old_ruleset:
        old_ruleset = ruleset
        ruleset = simplify_ruleset(ruleset, toplevel_rule_names)

    return ruleset

def get_follow_set_for_path(path, rules):
    if path.node:
        return path.node.get_follow_set_for_path(path, rules)
    else:
       return get_rule_follow_set(path.position, rules)

def get_rule_follow_set(rule_name, rules, rules_considered=None):
    if rules_considered is None:
        rules_considered = set()
    rules_considered.add(rule_name)

    result = set()
    # Go through all the rules, and find Nonterminals with `rule_name`.

    def handle_node(node):
        """Returns True if the follow set of `node` should be included in result"""
        match node:
            case Sequence(items):
                for pos, current in enumerate(items):
                    add_follow_set = False
                    match current:
                        case Nonterminal(n) if n == rule_name:
                            # we found Nonterminal with the value we're searching for
                            add_follow_set = True
                        case _:
                            if handle_node(current):
                                add_follow_set = True
                    if add_follow_set:
                        for following in items[pos+1:]:
                            start_tokens = following.get_possible_start_tokens(rules, set())
                            result.update(start_tokens - {None})
                            if None not in start_tokens:
                                break
                        else:
                            return True
            case Optional(item):
                return handle_node(item)
            case ZeroOrMore(item) | OneOrMore(item):
                if handle_node(item):
                    start_tokens = item.get_possible_start_tokens(rules, set())
                    result.update(start_tokens - {None})
                    return True
            case Choice(alts):
                include_follow = False
                for alt in alts:
                    if handle_node(alt):
                        include_follow = True
                return include_follow
            case Gather(sep, item):
                match sep:
                    case BaseToken():
                        pass
                    case _:
                        raise NotImplementedError()
                if handle_node(item):
                    start_tokens = sep.get_possible_start_tokens(rules, set())
                    result.update(start_tokens - {None})
                    if None in start_tokens:
                        raise NotImplementedError()
                    return True
            case Nonterminal(value):
                if value == rule_name:
                    return True
                else:
                    pass
            case BaseToken() | NegativeLookahead() | PositiveLookahead():
                pass
            case _:
                raise NotImplementedError(repr(node))

    for name, rule in rules.items():
        if handle_node(rule):
            if name not in rules_considered:
                result.update(get_rule_follow_set(
                    name, rules, rules_considered,
                ))

    return result

def generate_rule_lines(rules, rule_names, toplevel_rule_names, debug):
    # Figure out all rules we want to document.
    # This includes the ones we were asked to document (`rule_names`),
    # and also rules referenced from them (recursively), except ones that will
    # be documented elsewhere (those in `toplevel_rule_names`).
    requested_rule_names = list(rule_names)
    rule_names_to_consider = list(rule_names)
    rule_names_to_generate = []
    while rule_names_to_consider:
        rule_name = rule_names_to_consider.pop(0)
        if rule_name in rule_names_to_generate:
            continue
        rule_names_to_generate.append(rule_name)

        node = rules[rule_name]
        for descendant in generate_all_descendants(node, filter=Nonterminal):
            rule_name = descendant.value
            if (
                rule_name in rules
                and rule_name not in toplevel_rule_names
            ):
                rule_names_to_consider.append(rule_name)

    longest_name = max(rule_names_to_generate, key=len)
    available_space = 80 - len(longest_name) - len(' ::=  ')

    # Yield all the lines
    for name in rule_names_to_generate:
        node = rules[name]
        if debug:
            # To compare with pegen's stringification:
            yield f'{name} (repr): {node!r}'

        for num, line in enumerate(node.format_lines(available_space)):
            if num == 0:
                yield f'{name}: {line}'.rstrip()
            else:
                yield f'  : {line}'.rstrip()
        # rhs_line = node.format()
        # if isinstance(node, Choice) and len(rhs_line) > 40:
        #     # Present each alternative on its own line
        #     yield f'{name}:'
        #     for alt in node:
        #         yield f'  : | {alt.format()}'
        # else:
        #     yield f'{name}: {node.format()}'
        if debug:
            yield from node.dump_tree()

def simplify_node(rules, node, path):
    """Simplifies a node repeatedly until simplification no longer changes it"""
    last_node = None
    while node != last_node:
        last_node = node
        node = node.simplify(rules, path)
        # nifty debug output:
            # debug('simplified', last_node.format(), '->', node.format())
            # debug('from:')
            # for line in last_node.dump_tree():
            #     debug(line)
            # debug('to:')
            # for line in node.dump_tree():
            #     debug(line)
    return node


def simplify_ruleset(old_ruleset: dict, toplevel_rule_names):
    """Simplify and inline a bunch of rules"""

    # Generate new ruleset with simplified nodes
    new_ruleset = {}
    for rule_name, node in old_ruleset.items():
        new_ruleset[rule_name] = simplify_node(
            old_ruleset, node, PathEntry.root_path(rule_name),
        )

    # A rule will be inlined if we were not explicitly asked to provide a
    # definition for it (i.e. it is not in `toplevel_rule_names`), and:
    # - is only mentioned once, or
    # - its definition is short
    # - it expands to a single nonterminal

    # Count up all the references to rules we're documenting.
    reference_counts = collections.Counter()
    for node in new_ruleset.values():
        for descendant in generate_all_descendants(node, filter=Nonterminal):
            name = descendant.value
            if name in new_ruleset:
                reference_counts[name] += 1

    # Inline the rules we found
    for name, count in reference_counts.items():
        node = new_ruleset[name]
        if name not in toplevel_rule_names and (
            count == 1
            or len(node.format()) <= len(name) * 1.2
            or isinstance(node, Nonterminal)
        ):
            replaced_name = name
            replacement = node
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
