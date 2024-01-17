from pathlib import Path
import re
import argparse

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
argparser.add_argument("docs_dir", help="Directory with the docs. All .rst files in this (and subdirs) will be regenerated.")


# TODO: Document all these rules somewhere in the docs
FUTURE_TOPLEVEL_RULES = {'compound_stmt', 'simple_stmts', 'expression'}


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
    rules = dict(grammar.rules)

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
                        rules,
                        match[1].split(),
                        set(toplevel_rules) | FUTURE_TOPLEVEL_RULES,
                    ):
                        new_lines.append(f'   {line}\n')
                    new_lines.append('\n')
        while new_lines and not new_lines[-1].strip():
            del new_lines[-1]
        if original_lines != new_lines:
            print(f'Updating: {path}')
            with path.open(encoding='utf-8', mode='w') as file:
                file.writelines(new_lines)
        else:
            print(f'Unchanged: {path}')

def generate_rule_lines(rules, rule_names, toplevel_rule_names):
    rule_names = list(rule_names)
    seen_rule_names = set()
    while rule_names:
        rule_name = rule_names.pop(0)
        if rule_name in seen_rule_names:
            continue
        rule = rules[rule_name]
        yield rule
        seen_rule_names.add(rule_name)

        for descendant in generate_all_descendants(rule):
             if isinstance(descendant, pegen.grammar.NameLeaf):
                try:
                    referenced_rule = rules[descendant.value]
                except KeyError:
                    pass
                else:
                    if descendant.value not in toplevel_rule_names:
                        rule_names.append(descendant.value)

def generate_all_descendants(node):
    print(node, type(node))
    yield node
    for value in node:
        if isinstance(value, list):
            for child in value:
                yield from generate_all_descendants(child)
        else:
            yield from generate_all_descendants(value)

if __name__ == "__main__":
    main()
