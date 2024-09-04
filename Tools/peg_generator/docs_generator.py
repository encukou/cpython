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
    prog="docs_generator.py",
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
argparser.add_argument(
    "--image-dir",
    help="Directory into which diagrams are written. All .svg files in this "
        + "directory will be removed before new ones are generated. "
        + "Requires the railroad-diagrams library from PyPI.",
)


# TODO: Document all these rules somewhere in the docs
FUTURE_TOPLEVEL_RULES = set()


# Rules that should be replaced by another rule, *if* they have the
# same contents.
# (In the grammar, these rules may have different actions, which we ignore
# here.)
REPLACED_SYNONYMS = {
    'literal_expr': 'literal_pattern',
    't_primary': 'primary',  # TODO: add logic to tell that these are the same
    'value_pattern': 'attr',
}

# TODO:

# Do more inlining for the diagrams (we can preserve the names!)
# - OptionalSequence (for import)

# Add tests

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


# Ideas for extending the library:
# - Aligning
# - OptionalSequence with separators

def parse_docs(docs_dir):
    """
    Get all the top-level rule names, and the files we're updating
    "top-level" means a rule that the docs explicitly ask for in
    a `grammar-snippet` directive. Anything that references a top-level
    rule should link to it.
    """

    files_with_grammar = set()

    # Maps the name of a top-level rule to the path of the file it's in
    toplevel_rule_locations = {}

    # List of tuples of top-level rules that appear together
    snippet_rule_names = []

    for path in Path(docs_dir).glob('**/*.rst'):
        with path.open(encoding='utf-8') as file:
            for line in file:
                if match := HEADER_RE.fullmatch(line):
                    files_with_grammar.add(path)
                    names = tuple(match[1].split())
                    snippet_rule_names.append(names)
                    for name in names:
                        if name in toplevel_rule_locations:
                            raise ValueError(
                                f'rule {name!r} appears both in '
                                + f'{toplevel_rule_locations[name]} and in {path}. It '
                                + f'should only be documented in one place.')
                        toplevel_rule_locations[name] = path
    print(f'{toplevel_rule_locations=}')

    return files_with_grammar, snippet_rule_names


def main():
    args = argparser.parse_args()

    files_with_grammar, snippet_rule_names = parse_docs(args.docs_dir)

    grammar = Grammar.from_pegen_file(
        args.grammar_filename,
        snippet_rule_names,
        args.debug,
    )

    for path in files_with_grammar:
        update_file(path, grammar)

    if args.image_dir:
        generate_diagrams(grammar, args.image_dir)


def update_file(path, grammar):
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
                first_rule_name = match[1].split(maxsplit=1)[0]
                snippet = grammar.snippets[first_rule_name]
                for line in generate_rule_lines(snippet):
                    if line.strip():
                        new_lines.append(f'   {line}\n')
                    else:
                        new_lines.append('\n')
                new_lines.append('\n')

    while new_lines and not new_lines[-1].strip():
        del new_lines[-1]

    if original_lines != new_lines:
        print(f'Updating: {path}')
        with path.open(encoding='utf-8', mode='w') as file:
            file.writelines(new_lines)
    else:
        print(f'Unchanged: {path}')


class Grammar:
    """A representation of the complete grammar"""

    @classmethod
    def from_pegen_file(cls, grammar_filename, snippet_rule_names, debug=False):
        pegen_grammar, parser, tokenizer = build_parser(grammar_filename)
        pegen_rules = dict(pegen_grammar.rules)

        rules = {}
        for rule_name, pegen_rule in pegen_rules.items():
            node = convert_pegen_node(pegen_rule.rhs)
            rules[rule_name] = node

        self = cls(
            rules,
            snippet_rule_names,
            debug=debug,
        )

        if self.debug:
            for name, rule in self.rules.items():
                print()
                print(name, rule.format())
                for line in rule.dump_tree():
                    print(line)

        return self

    def __init__(self, rules, snippet_rule_names, debug=False):
        self.rules = rules
        self.debug = debug

        toplevel_rule_names = set()
        for names in snippet_rule_names:
            toplevel_rule_names.update(names)
        self.toplevel_rule_names = frozenset(toplevel_rule_names)

        self._simplify()

        self.snippets = {}
        for names in snippet_rule_names:
            snippet = Snippet(self, names)
            self.snippets[names[0]] = snippet

    def _simplify(self):
        for name, node in self.rules.items():
            path = PathEntry.root_path(name)
            self.rules[name] = node.simplify(self.rules, path)

        # Simplify the rules repeatedly,
        # until simplification no longer changes them
        old_rules = None
        while self.rules != old_rules:
            old_rules = self.rules
            self._simplify_ruleset()

    def _simplify_ruleset(self):
        """Simplify and inline a bunch of rules"""

        # Generate new ruleset with simplified nodes
        new_ruleset = {}
        for rule_name, node in self.rules.items():
            new_ruleset[rule_name] = node.simplify(
                self.rules, PathEntry.root_path(rule_name),
            )

        # A rule will be inlined if we were not explicitly asked to provide a
        # definition for it (i.e. it is not in `toplevel_rule_names`), and:
        # - is only mentioned once, or
        # - its definition is short
        # - it expands to a single nonterminal

        # Count up all the references to rules we're documenting.
        reference_counts = collections.Counter()
        for node in new_ruleset.values():
            for descendant in node.generate_descendants(filter_class=Nonterminal):
                name = descendant.value
                if name in new_ruleset:
                    reference_counts[name] += 1

        # Inline the rules we found
        for name, count in reference_counts.items():
            node = new_ruleset[name]
            if name not in self.toplevel_rule_names and (
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

        self.rules = new_ruleset


class Snippet:
    """Represents one group of rules in the documentation"""

    def __init__(self, grammar, initial_rule_names):
        self.grammar = grammar
        self.initial_rule_names = initial_rule_names

        self.documented_rules = {
            name: grammar.rules[name]
            for name in self._get_documented_rule_names()
        }

        self.rule_names_to_inline = frozenset(self._get_rule_names_to_inline())

    def _get_documented_rule_names(self):
        """Figure out all rules to include in this grammar snippet.

        This includes the ones we were asked to document (`initial_rule_names`),
        and also rules referenced from them (recursively), except ones that will
        be documented elsewhere (those in `self.toplevel_rule_names`).
        """
        grammar = self.grammar
        toplevel_rule_names = grammar.toplevel_rule_names | FUTURE_TOPLEVEL_RULES
        rule_names_to_consider = list(self.initial_rule_names)
        rule_names_to_generate = []
        while rule_names_to_consider:
            rule_name = rule_names_to_consider.pop(0)
            if rule_name in rule_names_to_generate:
                continue
            rule_names_to_generate.append(rule_name)

            node = grammar.rules[rule_name]
            for descendant in node.generate_descendants(filter_class=Nonterminal):
                rule_name = descendant.value
                if (
                    rule_name in grammar.rules
                    and rule_name not in toplevel_rule_names
                ):
                    rule_names_to_consider.append(rule_name)
        return rule_names_to_generate

    def _get_rule_names_to_inline(self):
        reference_counts = collections.Counter()
        for name, node in self.documented_rules.items():
            for descendant in node.generate_descendants(filter_class=Nonterminal):
                nonterminal_name = descendant.value
                if nonterminal_name in self.documented_rules and nonterminal_name != name:
                    reference_counts[nonterminal_name] += 1

        return frozenset(
            name for name, count in reference_counts.items() if count == 1
        )


# TODO: Check parentheses are correct in complex cases.

class Precedence(enum.IntEnum):
    CHOICE = enum.auto()
    SEQUENCE = enum.auto()
    REPEAT = enum.auto()
    LOOKAHEAD = enum.auto()
    ATOM = enum.auto()

class Node:
    def format(self) -> str:
        """Return self's representation, as a single-line string.
        """
        raise NotImplementedError()

    def __iter__(self):
        """Yield all child nodes."""
        raise NotImplementedError()

    def inlined(self, replaced_name, replacement):
        """Return a version of self with the given nonterminal replaced.

        For example, given:

            >>> str(node)
            number "+" number
            >>> str(replacement)
            digit*

        we should get:

            >>> inlined(node, "number", replacement)
            digit* "+" digit*
        """
        raise NotImplementedError()

    def get_possible_start_tokens(self, rules, rules_considered):
        """Return the set of tokens that strings that match self could start with.

        Additionally, if `None` is in the set, the node could match an empty
        string.

        The result is a Python set, which should be converted to TokenSet
        before use. See TokenSet for details.
        """
        raise NotImplementedError()

    def get_follow_set_for_path(self, path: 'PathEntry', rules: dict):
        """Return the set of tokens that can follow the child node
        identified by `path`.

        The result depends on parent nodes, which are given as `path`.

        The result is a Python set, which should be converted to TokenSet
        before use. See TokenSet for details.

        Unlike get_possible_start_tokens(), the result should not contain None.
        """
        raise NotImplementedError()

    def simplify(self, rules, path):
        """Simplify self repeatedly until simplification no longer changes it.

        This should not be overridden.
        """
        node = self
        last_node = None
        while node != last_node:
            last_node = node
            node = node.simplify_once(rules, path)
            # nifty debug output:
                # debug('simplified', last_node.format(), '->', node.format())
                # debug('from:')
                # for line in last_node.dump_tree():
                #     debug(line)
                # debug('to:')
                # for line in node.dump_tree():
                #     debug(line)
        return node

    def simplify_once(self, rules, path):
        """Return a simplified version of self."""
        return self

    def generate_descendants(self, filter_class=None):
        """Yield descendatnts of the given class, recursively.

        If filter_class is None, yield all descendants.
        """
        if filter_class is None or isinstance(self, filter_class):
            yield self
        for value in self:
            yield from value.generate_descendants(filter_class)

    def format_for_precedence(self, parent_precedence):
        """Like format(), but add parentheses if necessary.

        parent_precedence is the Precedence of the enclosing Node.
        """
        result = self.format()
        if self.precedence < parent_precedence:
            result = '(' + result + ')'
        return result

    def dump_tree(self, indent=0):
        """Yield lines of a debug representation of this node."""
        yield '  : ' + '  ' * indent + self.format()

    def format_lines(self, columns):
        """Yield lines of a string representation, as it should appear in docs.

        Concatenating the lines should be equivalent to calling format(),
        ignoring whitespace.
        """
        yield self.format()


@dataclass(frozen=True)
class Container(Node):
    """Collection of zero or more child items"""
    items: list[Node]

    def __iter__(self):
        yield from self.items

    def simplify_item(self, rules, item, path):
        return item.simplify(rules, path)

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
    """Grammar node that matches any of a sequence of alternatives."""
    precedence = Precedence.CHOICE

    def format(self):
        if not self.items:
            return '<UNREACHABLE>'
        return " | ".join(
            item.format_for_precedence(Precedence.CHOICE)
            for item in self
        )

    def simplify_once(self, rules, path):
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
            Sequence(alt).simplify(rules, path.child(self, None))
            for alt in alternatives
        ]))

    def simplify_subsequence(self, rules, subsequence):
        """Simplify the given subsequence of self's alternatives.
        """
        if len(subsequence) >= 2:
            # If two or more adjacent alternatives start or end with the
            # same item, we pull that item out, and replace the alternatives
            # with a sequence of
            # [common item, Choice of the remainders of the alts].
            first_alt = subsequence[0]
            # We do this for both the start and the end; for that we need the
            # index of the candidate item (0 or -1) and the slice to get the
            # rest of the items.
            for index, rest_slice in (
                    (0, slice(1, None)),
                    (-1, slice(None, -1)),
                ):
                num_alts_with_common_item = 1
                for alt in subsequence[1:]:
                    if alt[index] != first_alt[index]:
                        break
                    num_alts_with_common_item += 1
                if num_alts_with_common_item > 1:
                    common_item = first_alt[index]
                    remaining_choice = Choice([
                        Sequence(alt[rest_slice])
                        for alt in subsequence[:num_alts_with_common_item]
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

        match subsequence[:2]:
            case [
                [x],
                [Gather(_, x1), Optional()] as result,
            ] if x == x1:
                return [result], 2

        return [subsequence[0]], 1

    def get_possible_start_tokens(self, rules, rules_considered):
        result = set()
        for item in self.items:
            result.update(item.get_possible_start_tokens(rules, rules_considered))
        return result

    def get_follow_set_for_path(self, path, rules):
        return path.parent_entry.get_follow_set(rules)

@dataclass(frozen=True)
class Sequence(Container):
    precedence = Precedence.SEQUENCE

    def simplify_once(self, rules, path):
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
            result.update(path.parent_entry.get_follow_set(rules))
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

    def simplify_once(self, rules, path):
        self_type = type(self)
        item = self.item.simplify(rules, path.child(self))
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

    def simplify_once(self, rules, path):
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
        return super().simplify_once(rules, path)

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
            | path.parent_entry.get_follow_set(rules)
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
            | path.parent_entry.get_follow_set(rules)
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
        the lookahead can match a string longer than a single token."""
        item = self.item
        if isinstance(item, Nonterminal):
            item = rules[item.value]
        if isinstance(item, Choice):
            tokens_in_self = item.items
        else:
            tokens_in_self = {item}
        if not all(isinstance(item, BaseToken) for item in tokens_in_self):
            # we only simplify lookaheads that only contain single tokens
            return None
        return TokenSet(tokens_in_self)


class NegativeLookahead(Lookahead):
    sigil = '!'

    def simplify_once(self, rules, path):
        # Find all the tokens the lookahead contains
        # (We don't simplify lookaheads that contain more than tokens)
        tokens_in_self = self.tokens_in_self(rules)
        if tokens_in_self:
            self_follow_set = TokenSet(path.get_follow_set(rules))

            if not tokens_in_self.issubset(self_follow_set):
                # no tokens in the neg.lookahead can follow it,
                # so the lookahead is redundant
                return EMPTY
        return super().simplify_once(rules, path)

class PositiveLookahead(Lookahead):
    sigil = '&'

    def simplify_once(self, rules, path):
        # Find all the tokens the lookahead contains
        # (We don't simplify lookaheads that contain more than tokens)
        tokens_in_self = self.tokens_in_self(rules)
        if tokens_in_self:
            self_follow_set = TokenSet(path.get_follow_set(rules))

            if tokens_in_self.issuperset(self_follow_set):
                # no other tokens than what's in the lookahead can follow it,
                # so the lookahead is redundant
                return EMPTY
        return super().simplify_once(rules, path)


@dataclass(frozen=True)
class Gather(Node):
    separator: Node
    item: Node

    precedence = Precedence.REPEAT

    def __iter__(self):
        yield self.separator
        yield self.item

    def simplify_once(self, rules, path):
        self_type = type(self)
        return self_type(
            self.separator.simplify(rules, path.child(self, 'sep')),
            self.item.simplify(rules, path.child(self, 'item')),
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
                        path.parent_entry.get_follow_set(rules)
                    )
                    result.discard(None)
                return result
            case 'item':
                result = set(sep_start_tokens)
                if None in result:
                    result.update(item_start_tokens)
                    result.discard(None)
                result.update(path.parent_entry.get_follow_set(rules))
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
    """A token, also known as a terminal"""
    def get_possible_start_tokens(self, rules, rules_considered):
        return {self}

class LiteralToken(BaseToken):
    """A token that matches an exact string, like "_" or ','or 'if'
    """
    @property
    def kind(self):
        """The kind of self, as a value for SymbolicToken"""
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
    """A reference to a named rule."""
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

    def simplify_once(self, rules, path):
        if other_rule_name := REPLACED_SYNONYMS.get(self.value):
            self_rule = rules[self.value]
            other_rule = rules[other_rule_name]
            if self_rule == other_rule:
                return Nonterminal(other_rule_name)
            if self_rule == Nonterminal(other_rule_name):
                return Nonterminal(other_rule_name)
        return super().simplify_once(rules, path)

EMPTY = Node.EMPTY = Sequence([])
UNREACHABLE = Node.UNREACHABLE = Choice([])

def convert_pegen_node(pegen_node):
    """Convert a pegen grammar node to our AST node."""
    match pegen_node:
        case pegen.grammar.Rhs():
            return Choice([convert_pegen_node(alt) for alt in pegen_node.alts])
        case pegen.grammar.Alt():
            if 'RAISE_SYNTAX_ERROR' in (pegen_node.action or ''):
                # This is actually invalid syntax,
                # see https://github.com/python/cpython/issues/118235
                return UNREACHABLE
            return Sequence([convert_pegen_node(item) for item in pegen_node.items])
        case pegen.grammar.Group():
            return convert_pegen_node(pegen_node.rhs)
        case pegen.grammar.NamedItem():
            return convert_pegen_node(pegen_node.item)
        case pegen.grammar.Opt():
            return Optional(convert_pegen_node(pegen_node.node))
        case pegen.grammar.NameLeaf():
            if pegen_node.value == 'TYPE_COMMENT':
                # The tokenizer doesn't emit TYPE_COMMENT unless it's in
                # a special mode
                return UNREACHABLE
            if pegen_node.value.isupper():
                return SymbolicToken(pegen_node.value)
            if pegen_node.value.startswith('invalid'):
                return UNREACHABLE
            else:
                return Nonterminal(pegen_node.value)
        case pegen.grammar.StringLeaf():
            return LiteralToken(pegen_node.value)
        case pegen.grammar.Repeat1():
            return OneOrMore(convert_pegen_node(pegen_node.node))
        case pegen.grammar.Repeat0():
            return ZeroOrMore(convert_pegen_node(pegen_node.node))
        case pegen.grammar.PositiveLookahead():
            return PositiveLookahead(convert_pegen_node(pegen_node.node))
        case pegen.grammar.NegativeLookahead():
            return NegativeLookahead(convert_pegen_node(pegen_node.node))
        case pegen.grammar.Cut():
            return EMPTY
        case pegen.grammar.Forced():
            return convert_pegen_node(pegen_node.node)
        case pegen.grammar.Gather():
            return Gather(
                convert_pegen_node(pegen_node.separator),
                convert_pegen_node(pegen_node.node),
            )
        case _:
            raise TypeError(f'{pegen_node!r} has unknown type {type(pegen_node).__name__}')

@dataclass(frozen=True)
class PathEntry:
    """Represents a path through the grammar tree from the root to a given node.

    The path can be traversed by following `parent_entry` attributes.

    Each entry has an associated `node`, and a `position` within the parent.
    The type and meaning of `position` depend on the kind of `node`.

    For example, given a rule like:

        number := ["-"] digits+

    the PathEntry for "digits" would be:

        PathEntry(
            parent_entry=PathEntry(
                parent_entry=PathEntry(
                    node=None,          # the root -- the entire grammar
                    position="number",  # name of the rule
                    parent_entry=None,
                ),
                node=Sequence(...),     # the whole sequence: `["-"] digits+`
                position = 2,           # the position in the sequence
            ),
            node=OneOrMore(...),        # `digits+`
            position=None,              # OneOrMore always has one child,
                                        # no position needed
        )
    """
    parent_entry: typing.Union["PathEntry", None]
    node: Node | None
    position: object

    def child(self, node, position=None):
        """Create a child PathEntry"""
        return PathEntry(self, node, position)

    @classmethod
    def root_path(cls, rule_name):
        return cls(None, None, rule_name)

    def get_follow_set(self, rules):
        """Return the set of tokens that can follow the node identified by `path`.

        The result is a Python set, which should be converted to TokenSet
        before use. See TokenSet for details.
        """
        if self.node:
            return self.node.get_follow_set_for_path(self, rules)
        else:
            # A root path identifies a rule rather than a node
            rule_name = self.position
            return get_rule_follow_set(rule_name, rules)


class TokenSet:
    """A set of tokens.

    TokenSet's operations take into account that
    `SymbolicToken`s like NAME are actually sets themselves.
    For example, NAME can match one of an infinite set of tokens:
    "a", "b", "c", ... "aa", "ab", ...

    Only the needed methods from `set`'s interface are implemented :)
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

def get_rule_follow_set(rule_name, rules, rules_considered=None):
    """Return a set of all tokens that might follow the given rule"""
    if rules_considered is None:
        rules_considered = set()
    rules_considered.add(rule_name)

    # Go through all the rules, and find Nonterminals with `rule_name`.

    def handle_node(node):
        """Return a set of tokens that should be included in rhe result.

        TODO:
        Additionally, if None is in the returned set, the the follow set of
        `node` should be also added to the result
        """

        match node:
            case Sequence(items):
                result = set()
                for pos, current in enumerate(items):
                    add_follow_set = False
                    match current:
                        case Nonterminal(n) if n == rule_name:
                            # we found Nonterminal with the value we're searching for
                            add_follow_set = True
                        case _:
                            include_follow, tokens_to_include = handle_node(current)
                            result.update(tokens_to_include)
                            if include_follow:
                                add_follow_set = True
                    if add_follow_set:
                        for following in items[pos+1:]:
                            start_tokens = following.get_possible_start_tokens(rules, set())
                            result.update(start_tokens - {None})
                            if None not in start_tokens:
                                break
                        else:
                            return True, result
                return False, result
            case Optional(item):
                return handle_node(item)
            case ZeroOrMore(item) | OneOrMore(item):
                include_follow, tokens_to_include = handle_node(item)
                if include_follow:
                    start_tokens = item.get_possible_start_tokens(rules, set())
                    return True, tokens_to_include | start_tokens - {None}
                return False, tokens_to_include
            case Choice(alts):
                include_follow = False
                result = set()
                for alt in alts:
                    include_follow_here, tokens_to_include = handle_node(alt)
                    result.update(tokens_to_include)
                    if include_follow_here:
                        include_follow = True
                return include_follow, result
            case Gather(separator, item):
                match separator:
                    case BaseToken():
                        pass
                    case _:
                        raise NotImplementedError()
                include_follow, tokens_to_include = handle_node(item)
                if include_follow:
                    start_tokens = separator.get_possible_start_tokens(rules, set())
                    if None in start_tokens:
                        raise NotImplementedError()
                    return True, tokens_to_include | start_tokens - {None}
                return False, tokens_to_include
            case Nonterminal(value):
                if value == rule_name:
                    return True, set()
                else:
                    return False, set()
            case BaseToken() | NegativeLookahead() | PositiveLookahead():
                return False, set()
            case _:
                raise NotImplementedError(repr(node))

    result = set()
    for name, rule in rules.items():
        include_follow, tokens_to_include = handle_node(rule)
        result.update(tokens_to_include)
        if include_follow:
            if name not in rules_considered:
                result.update(get_rule_follow_set(
                    name, rules, rules_considered,
                ))
    return result


def generate_rule_lines(snippet):
    grammar = snippet.grammar
    rule_names_to_generate = snippet.documented_rules

    diagram_names = [
        name for name in snippet.documented_rules
        if name not in snippet.rule_names_to_inline
    ]

    yield ':group: python-grammar'
    yield f':generated-by: {SCRIPT_NAME}'
    yield f':diagrams: {' '.join(diagram_names)}'
    yield ''

    longest_name = max(snippet.documented_rules, key=len)
    available_space = 80 - len(longest_name) - len(' ::=  ')

    # Yield all the lines
    for name, node in snippet.documented_rules.items():
        if grammar.debug:
            # To compare with pegen's stringification:
            yield f'{name} (repr): {node!r}'

        for num, line in enumerate(node.format_lines(available_space)):
            if num == 0:
                yield f'{name}: {line}'.rstrip()
            else:
                yield f'  : {line}'.rstrip()
        # TODO: fold long lines
        # rhs_line = node.format()
        # if isinstance(node, Choice) and len(rhs_line) > 40:
        #     # Present each alternative on its own line
        #     yield f'{name}:'
        #     for alt in node:
        #         yield f'  : | {alt.format()}'
        # else:
        #     yield f'{name}: {node.format()}'
        if grammar.debug:
            yield from node.dump_tree()


def node_to_diagram_element(railroad, node, rules, rules_to_inline):
    def recurse(node, rules_to_inline):
        match node:
            case Sequence(children):
                return railroad.Sequence(*(recurse(c, rules_to_inline) for c in children))
            case Choice(children):
                return railroad.Choice(0, *(recurse(c, rules_to_inline) for c in children))
            case Optional(child):
                return railroad.Optional(recurse(child, rules_to_inline))
            case ZeroOrMore(child):
                return railroad.ZeroOrMore(recurse(child, rules_to_inline))
            case OneOrMore(child):
                return railroad.OneOrMore(recurse(child, rules_to_inline))
            case Gather(sep, item):
                return railroad.OneOrMore(
                    recurse(item, rules_to_inline),
                    recurse(sep, rules_to_inline),
                )
            case PositiveLookahead(child):
                return railroad.Group(
                    recurse(child, rules_to_inline),
                    "lookahead",
                )
            case NegativeLookahead(child):
                return railroad.Group(
                    recurse(child, rules_to_inline),
                    "negative lookahead",
                )
            case Nonterminal(name):
                if name in rules_to_inline and name in rules_to_inline:
                    rules_to_inline = rules_to_inline - {name}
                    inlined_diagram = recurse(rules[name], rules_to_inline)
                    return railroad.Group(inlined_diagram, name)
                return railroad.NonTerminal(name)
            case SymbolicToken(name):
                return railroad.Terminal(name)
            case LiteralToken(name):
                return railroad.Terminal(name)
            case _:
                raise ValueError(node)
    return recurse(node, rules_to_inline)


def generate_diagrams(grammar, image_dir):
    import railroad

    path = Path(image_dir)
    try:
        path.mkdir()
    except FileExistsError:
        for old_path in path.glob('*.svg'):
            old_path.unlink()
    for snippet in grammar.snippets.values():
        rules_to_inline = snippet.rule_names_to_inline.intersection(snippet.documented_rules)

        for name, node in snippet.documented_rules.items():
            if name.startswith("invalid_"):
                continue
            dest_path = path / f'{name}.svg'
            print(f'Generating: {dest_path}')
            d = railroad.Diagram(
                node_to_diagram_element(railroad, node, grammar.rules, rules_to_inline),
                type='simple',
            )
            with open(dest_path, 'w') as file:
                d.writeStandalone(file.write)


if __name__ == "__main__":
    main()
