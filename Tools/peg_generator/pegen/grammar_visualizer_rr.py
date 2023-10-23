import argparse
import sys
from typing import Any, Callable, Iterator
import unicodedata

from pegen.build import build_parser
from pegen.python_generator import InvalidNodeVisitor
from pegen.grammar import Grammar, Rule, Rhs, Alt, NamedItem, StringLeaf
from pegen.grammar import NameLeaf, GrammarVisitor

argparser = argparse.ArgumentParser(
    prog="pegen", description="Print a given PEG grammar as railroad diagrams"
)
argparser.add_argument("filename", help="Grammar description")

def get_components(char):
    name = unicodedata.name(char).removeprefix('BOX DRAWINGS ')
    result = []
    for part in name.split(' AND '):
        part = part.replace(' DASH', '-DASH')
        directions = []
        for word in part.split():
            match word:
                case 'LIGHT' | 'HEAVY':
                    style = word
                case ('DOUBLE' | 'DOUBLE-DASH' | 'TRIPLE-DASH'
                      | 'QUADRUPLE-DASH'):
                    style = word
                case 'SINGLE':
                    style = 'LIGHT'
                case 'DASH':
                    pass
                case 'LEFT' | 'RIGHT' | 'UP' | 'DOWN':
                    directions.append(word)
                case 'HORIZONTAL':
                    directions.extend(['LEFT', 'RIGHT'])
                case 'VERTICAL':
                    directions.extend(['UP', 'DOWN'])
                case 'ARC' | 'DIAGONAL':
                    # We don't compose these
                    return None
                case _:
                    raise ValueError(word)
        result.extend((style, d) for d in directions)
    return frozenset(result)

BOX_CHARS = tuple({chr(n) for n in range(0x2500, 0x2580)})
BOX_CHARS_BY_COMPONENT = {get_components(c): c  for c in BOX_CHARS}
del BOX_CHARS_BY_COMPONENT[None]
BOX_CHARS_COMPONENTS = {v: k for k, v in BOX_CHARS_BY_COMPONENT.items()}

class Canvas:
    def __init__(self, contents=()):
        self.contents = {}
        if contents:
            self.update(contents)
        self._bounds = None

    def __setitem__(self, row_col, symbol):
        row, col = row_col
        self.contents[row_col] = symbol
        self._bounds = None

    def update(self, items):
        if isinstance(items, Canvas):
            self.update(items.contents)
        else:
            for coords, char in items.items():
                row, col = coords
                if ((c1 := BOX_CHARS_COMPONENTS.get(char))
                    and (c2 := BOX_CHARS_COMPONENTS.get(
                         self.contents.get(coords)))):
                    self[coords] = BOX_CHARS_BY_COMPONENT.get(c1 | c2, char)
                else:
                    self[coords] = char
            self._bounds = None

    def blit(self, other, row, col):
        self.update({
            (row+r, col+c): char
            for (r, c), char in other.contents.items()
        })

    def shift(self, row, col):
        self.contents = {
            (row+r, col+c): char
            for (r, c), char in self.contents.items()
        }
        self._bounds = None

    def draw_text(self, message, row=0, col=0):
        for c, char in enumerate(message):
            self[row, col + c] = char

    def draw_hline(self, row, col1, col2, style='LIGHT', chars=None):
        if chars is None:
            chars = (
                BOX_CHARS_BY_COMPONENT[frozenset({(style, 'RIGHT')})]
                + BOX_CHARS_BY_COMPONENT[frozenset({(style, 'LEFT'), (style, 'RIGHT')})]
                + BOX_CHARS_BY_COMPONENT[frozenset({(style, 'LEFT')})]
            )

        self.update({
            (row, col1): chars[0],
            **{(row, c): chars[1] for c in range(col1 + 1, col2)},
            (row, col2): chars[2],
        })

    def draw_vline(self, row1, row2, col, style='LIGHT', chars=None):
        if chars is None:
            chars = (
                BOX_CHARS_BY_COMPONENT[frozenset({(style, 'DOWN')})]
                + BOX_CHARS_BY_COMPONENT[frozenset({(style, 'DOWN'), (style, 'UP')})]
                + BOX_CHARS_BY_COMPONENT[frozenset({(style, 'UP')})]
            )

        self.update({
            (row1, col): chars[0],
            **{(r, col): chars[1] for r in range(row1 + 1, row2)},
            (row2, col): chars[2],
        })

    def draw_box(self, row1, col1, row2, col2, style='LIGHT', chars=None):
        if chars is None:
            self.draw_hline(row1, col1, col2, style)
            self.draw_hline(row2, col1, col2, style)
            self.draw_vline(row1, row2, col1, style)
            self.draw_vline(row1, row2, col2, style)
        else:
            self.draw_hline(row1, col1, col2, chars=chars[0:3])
            self.draw_hline(row2, col1, col2, chars=chars[6:9])
            self.draw_vline(row1+1, row2-1, col1, chars=chars[3]*3)
            self.draw_vline(row1+1, row2-1, col2, chars=chars[5]*3)

    def enclose(self, style='LIGHT', chars=None):
        self.draw_box(
            self.min_row-1, self.min_col-1,
            self.max_row+1, self.max_col+1,
            style, chars,
        )

    def _get_bounds(self):
        if self._bounds is None:
            if self.contents:
                rows = sorted(set(row for row, col in self.contents))
                cols = sorted(set(col for row, col in self.contents))
                self._bounds = rows[0], cols[0], rows[-1], cols[-1]
            else:
                self._bounds = 0, 0, 0, 0
        return self._bounds

    @property
    def min_row(self):
        return self._get_bounds()[0]
    @property
    def min_col(self):
        return self._get_bounds()[1]
    @property
    def max_row(self):
        return self._get_bounds()[2]
    @property
    def max_col(self):
        return self._get_bounds()[3]
    @property
    def height(self):
        bounds = self._get_bounds()
        return bounds[2] - bounds[0]
    @property
    def width(self):
        bounds = self._get_bounds()
        return bounds[3] - bounds[1]

    def normalized(self):
        min_row = self.min_row
        min_col = self.min_col
        if min_row == min_col == 0:
            return self
        return Canvas({(r - min_row, c - min_col): char
                       for (r, c), char in self.contents.items()})

    def dump(self):
        for row in range(self.min_row, self.max_row + 1):
            for col in range(self.min_col, self.max_col + 1):
                print(self.contents.get((row, col), ' '), end='')
            print()

class RailroadVisitor(GrammarVisitor):
    def visit_Grammar(self, node):
        for rule in node:
            if not rule.name.startswith('invalid_'):
                yield self.visit(rule)

    def visit_Rule(self, node):
        canvas = self.visit(*node)
        canvas.draw_text('►►', 0, canvas.min_col-2)
        canvas.draw_text('►◄', 0, canvas.max_col+1)
        canvas.draw_text(node.name + ':', canvas.min_row - 1, canvas.min_col)
        return canvas

    def visit_Rhs(self, node):
        canvases = [self.visit(child) for child in node.alts]
        if len(canvases) == 1:
            return canvases[0]
        max_width = max(c.width for c in canvases)
        result = Canvas()
        result.draw_hline(0, -1, 1)
        result.draw_hline(0, max_width, max_width + 2)
        current_row = first_row = last_row = canvases[0].min_row
        for canvas in canvases:
            current_row -= canvas.min_row
            result.blit(canvas, current_row, -canvas.min_col)
            result.draw_hline(current_row, canvas.width, max_width + 1)
            last_row = current_row
            current_row += canvas.max_row + 1
        result.draw_vline(0, last_row, 0)
        result.draw_vline(0, last_row, max_width + 1)
        return result

    def visit_Alt(self, node):
        canvases = [self.visit(child) for child in node.items]
        return self._layout_horizontally(canvases)

    def _layout_horizontally(self, canvases):
        result = Canvas()
        current_col = 2
        for canvas in canvases:
            result.draw_hline(0, current_col - 1, current_col)
            new_col = current_col + canvas.width + 2
            result.draw_hline(0, new_col - 2, new_col - 1)
            result.blit(canvas, 0, current_col - canvas.min_col)
            current_col = new_col
        return result

    def visit_NamedItem(self, node):
        [child] = node
        return self.visit(child)
    visit_Group = visit_NamedItem

    def visit_NameLeaf(self, node):
        result = Canvas()
        result.draw_text(node.value)
        result.enclose('HEAVY')
        return result

    def visit_StringLeaf(self, node):
        result = Canvas()
        result.draw_text(node.value)
        result.enclose(chars='╭─╮│ │╰─╯')
        return result

    def visit_PositiveLookahead(self, node):
        canvas = self.visit(node.node)
        canvas.draw_text('lookahead', canvas.min_row - 1, canvas.min_col)
        canvas.enclose(chars='╭─╮┆ ┆╰┄╯')
        canvas.shift(1 - canvas.min_row, -canvas.min_col)
        canvas.draw_hline(0, canvas.min_col, canvas.max_col)
        canvas.draw_vline(0, 1, 2)
        canvas.draw_text('&', 0, 2)
        return canvas

    def visit_NegativeLookahead(self, node):
        canvas = self.visit(node.node)
        canvas.draw_text('neg.lookahead', canvas.min_row - 1, canvas.min_col)
        canvas.enclose(chars='╭─╮┆ ┆╰┄╯')
        canvas.shift(1 - canvas.min_row, -canvas.min_col)
        canvas.draw_hline(0, canvas.min_col, canvas.max_col)
        canvas.draw_vline(0, 1, 2)
        canvas.draw_text('!', 0, 2)
        return canvas

    def visit_Opt(self, node):
        canvas = self._make_opt(self.visit(node.node))
        #canvas.draw_text('?', 0, canvas.max_col-1)
        return canvas

    def _make_opt(self, canvas):
        canvas.shift(1, 0)
        canvas.draw_hline(canvas.min_row-1, canvas.min_col-1, canvas.max_col+1)
        canvas.draw_vline(canvas.min_row, 1, canvas.min_col)
        canvas.draw_vline(canvas.min_row, 1, canvas.max_col)
        canvas.draw_hline(1, canvas.min_col, canvas.min_col+1)
        canvas.draw_hline(1, canvas.max_col-1, canvas.max_col)
        return canvas

    def visit_Gather(self, node):
        canvas = self.visit(node.node)
        prev_min_col = canvas.min_col
        prev_max_col = canvas.max_col
        sep_canvas = self.visit(node.separator)
        canvas.draw_vline(0, -sep_canvas.min_row+1, canvas.max_col+1)
        canvas.blit(
            sep_canvas,
            -sep_canvas.min_row+1,
            canvas.max_col-sep_canvas.min_col+1,
        )
        canvas.draw_hline(0, prev_max_col, canvas.max_col+1)
        canvas.draw_hline(-sep_canvas.min_row+1, prev_max_col+1, prev_max_col+2)
        canvas.draw_hline(-sep_canvas.min_row+1, canvas.max_col-1, canvas.max_col)
        canvas.draw_vline(-sep_canvas.min_row+1, canvas.max_row+1, canvas.max_col)
        canvas.draw_hline(canvas.max_row, canvas.min_col-1, canvas.max_col)
        canvas.draw_vline(0, canvas.max_row, canvas.min_col)
        canvas.draw_hline(0, canvas.min_col, prev_min_col)
        canvas.draw_text('↑', 1, canvas.min_col)
        return canvas

    def visit_Cut(self, node):
        canvas = Canvas()
        canvas.draw_vline(0, 2, 0)
        canvas.draw_text('~')
        canvas.draw_text('cut', 2, -1)
        return canvas

    def visit_Repeat1(self, node):
        canvas = self._make_rep1(self.visit(node.node))
        #canvas.draw_text('+', -1, canvas.max_col-1)
        return canvas

    def _make_rep1(self, canvas):
        canvas.draw_hline(0, canvas.max_col, canvas.max_col+2)
        canvas.draw_vline(0, canvas.max_row+1, canvas.max_col-1)
        canvas.draw_hline(0, canvas.min_col-2, canvas.min_col)
        canvas.draw_vline(0, canvas.max_row, canvas.min_col+1)
        canvas.draw_hline(canvas.max_row, canvas.min_col+1, canvas.max_col-1)
        canvas.draw_text('↑', 1, canvas.min_col+1)
        return canvas

    def visit_Repeat0(self, node):
        canvas = self._make_opt(self._make_rep1(self.visit(node.node)))
        #canvas.draw_text('*', 0, canvas.max_col-1)
        return canvas

    def visit_Forced(self, node):
        canvas = self.visit(node.node)
        max_col = canvas.max_col
        shift = -canvas.min_row+1
        canvas.shift(shift, 0)
        canvas.draw_text('&&forced ', 0, 0)
        canvas.draw_vline(0, shift, canvas.min_col-1)
        canvas.draw_vline(0, shift, canvas.max_col+1)
        canvas.draw_hline(shift, canvas.min_col, canvas.min_col+1)
        canvas.draw_hline(shift, max_col, canvas.max_col)
        return canvas

    def generic_visit(self, node) -> Any:
        canvas = Canvas()
        canvas.draw_text('?<' + repr(node) + '>')
        canvas.enclose(chars='\U0001fba3\U0001fba6\U0001fba2\U0001fba4 \U0001fba5\U0001fba1\U0001fba7\U0001fba0')
        return canvas

class CleanupVisitor(GrammarVisitor):
    def visit_Grammar(self, node):
        return Grammar(
            rules=[
                self.visit(rule) for rule in node
                if not rule.name.startswith('invalid_')
            ],
            metas=node.metas,
        )

    def visit_Rhs(self, node):
        return Rhs(node.alts)

    def visit_Alt(self, node):
        return Alt([self.visit(a) for a in node])

    def generic_visit(self, node):
        return node


def main():
    args = argparser.parse_args()

    grammar, parser, tokenizer = build_parser(args.filename)

    canvases = RailroadVisitor().visit(grammar)
    for canvas in canvases:
        canvas.dump()


if __name__ == "__main__":
    main()
