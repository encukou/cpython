import argparse
import sys
from typing import Any, Callable, Iterator

from railroad import Diagram, Choice, DEFAULT_STYLE, Sequence, NonTerminal
from railroad import Terminal, Optional, ZeroOrMore, OneOrMore, Group

from pegen.build import build_parser
from pegen.grammar import Grammar, Rule, GrammarVisitor

argparser = argparse.ArgumentParser(
    prog="pegen", description="Pretty print the AST for a given PEG grammar"
)
argparser.add_argument("filename", help="Grammar description")

class DiagramVisitor(GrammarVisitor):
    def generic_visit(self, node):
        raise ValueError(node)

    def visit_Rule(self, node):
        return self.visit(node.rhs)

    def visit_Rhs(self, node):
        diagrams = (self.visit(n) for n in node.alts)
        return Choice(0, *(diagram for diagram in diagrams if diagram))

    def visit_Alt(self, node):
        diagrams = (self.visit(n) for n in node.items)
        return Sequence(*(diagram for diagram in diagrams if diagram))

    def visit_NamedItem(self, node):
        return self.visit(node.item)

    def visit_NameLeaf(self, node):
        return NonTerminal(node.value)

    def visit_StringLeaf(self, node):
        return Terminal(node.value)

    def visit_Opt(self, node):
        diagram = self.visit(node.node)
        if diagram is None:
            return None
        return Optional(diagram)

    def visit_Repeat0(self, node):
        diagram = self.visit(node.node)
        if diagram is None:
            return None
        return ZeroOrMore(diagram)

    def visit_Repeat1(self, node):
        diagram = self.visit(node.node)
        if diagram is None:
            return None
        return OneOrMore(diagram)

    def visit_NegativeLookahead(self, node):
        # We ignore lookaheads in docs. If we didn't, we could return:
        # Group(self.visit(node.node), 'negative lookahead')
        return None

    def visit_PositiveLookahead(self, node):
        # See visit_NegativeLookahead
        return None

    def visit_Gather(self, node):
        diagram = self.visit(node.node)
        if diagram is None:
            raise NotImplementedError('Gather element would be ignored')
        sep_diagram = self.visit(node.separator)
        if sep_diagram is None:
            return OneOrMore(diagram)
        return OneOrMore(diagram, sep_diagram)

    def visit_Group(self, node):
        return self.visit(node.rhs)

    def visit_Forced(self, node):
        # We render as if it wasn't forced
        diagram = self.visit(node.node)
        if diagram is None:
            return None
        return Group(diagram, 'forced')

    def visit_Cut(self, node):
        return Group([], 'cut')

def generate_svg(grammar):
    with open('output.html', 'w') as file:
        file.write("<html><head><style>")
        file.write(DEFAULT_STYLE)
        file.write("</style><body>")
        for rule in grammar:
            if rule.name.startswith('invalid_'):
                continue
            diagram = svg_from_rule(rule)
            file.write(f"<div><code>{rule.name}:</code><div>")
            diagram.writeSvg(file.write)
            print(rule)
        file.write("</body></html>")

def svg_from_rule(rule):
    visitor = DiagramVisitor()
    d = Diagram(visitor.visit(rule))
    return d


def main() -> None:
    args = argparser.parse_args()

    try:
        grammar, parser, tokenizer = build_parser(args.filename)
    except Exception as err:
        print("ERROR: Failed to parse grammar file", file=sys.stderr)
        sys.exit(1)

    generate_svg(grammar)


if __name__ == "__main__":
    main()
