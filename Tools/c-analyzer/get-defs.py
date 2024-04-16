from functools import lru_cache, total_ordering, cached_property
from collections import namedtuple, ChainMap
from dataclasses import dataclass, field
from contextlib import contextmanager
from pathlib import Path
import subprocess
import sysconfig
import string
import shlex
import enum
import ast
import sys
import re

# https://en.wikipedia.org/wiki/ANSI_escape_code#CSI_(Control_Sequence_Introducer)_sequences
CSI_RE = re.compile(r'\033\[[0-?]*[ -/]*[@-~]')

def csi(code):
    result = '\033' + code
    assert CSI_RE.fullmatch(result)
    return result

INTENSE = csi('[1m')
DIM = csi('[2m')
RED = csi('[1;31m')
CYAN = csi('[36m')
RESET = csi('[m')

getvar = sysconfig.get_config_var

if getvar('CC') != 'gcc':
    raise Exception('This script currently works with gcc only. Ports welcome.')

CFLAGS = getvar('CFLAGS')

projectpath = Path('.')
while not projectpath.joinpath('Include').exists():
    if projectpath == projectpath.parent:
        raise Exception('Could not find project root')
    projectpath = projectpath.parent

@dataclass(frozen=True)
class Location:
    filename: str
    lineno: int

    def __repr__(self):
        return f'<{self}>'

    def __str__(self):
        return f'{self.filename}:{self.lineno}'

@lru_cache
@total_ordering
@dataclass(frozen=True)
class Variant:
    limited_api: str

    def cflags(self):
        if self.limited_api is None:
            return []
        else:
            return [f'-DPy_LIMITED_API={hex(self.limited_api)}']

    def __str__(self):
        if self.limited_api:
            if self.limited_api == 3:
                return f'L3'
            else:
                major = (self.limited_api & 0xFF000000) >> 24
                minor = (self.limited_api & 0x00FF0000) >> 16
                rest = self.limited_api & ~0xFFFF0000
                if not rest:
                    return f'L{major}.{minor}'
                else:
                    return f'L{self.limited_api:x}'
        else:
            return 'cpy'

    def __lt__(self, other):
        return (self.limited_api or 0) < (other.limited_api or 0)

    @cached_property
    def next(self):
        if self.limited_api == 3:
            return Variant(limited_api=0x03020000)
        elif self.limited_api:
            major = (self.limited_api & 0xFF000000) >> 24
            minor = (self.limited_api & 0x00FF0000) >> 16
            rest = self.limited_api & ~0xFFFF0000
            return Variant(limited_api=(major << 24) | ((minor+1) << 16) | rest)

    def __repr__(self):
        return f'<{self}>'

@dataclass
class Definition:
    kind: str
    name: str
    args: str
    variants: dict[Variant, Location] = field(default_factory=dict, compare=False)

    def __str__(self):
        return CSI_RE.sub('', self.colorized)

    @property
    def colorized(self):
        return repr(self)


@dataclass
class MacroDef(Definition):
    body: str = ''

    @property
    def colorized(self):
        return f'#define {INTENSE}{self.name}{RESET}{self.args or ''} {self.body}'

@dataclass
class MacroUndef(Definition):
    @property
    def colorized(self):
        return f'#undef {DIM}{self.name}{RESET}{self.args or ''}'


@dataclass
class Entry:
    name: str
    definitions: list[Definition]

C_TOKEN_RE = re.compile(r'''
    # white
      \s+
    # identifier
    | [_a-zA-Z]
      [_a-zA-Z0-9]+
    # number
    | \.?
      [0-9]+
      ([a-zA-Z0-9_.] | [eEpP][+-])*
    # string
    | ["] ([^"] | \\["])+ ["]
    | ['] ([^'] | \\['])+ [']
    # punct
    | [-][>] | [+][+] | [-][-] | [<][<] | [>][>] | [<][=] | [>][=] | [=][=]
    | [!][=] | [&][&] | [|][|] | [*][=] | [/][=] | [%][=] | [+][=] | [-][=]
    | [&][=] | [^][=] | [|][=] | [#][#]
    | [.][.][.] | [<][<][=] | [>][>][=]
    | [<][:] | [:][>] | [<][%] | [%][>] | [%][:] | [%][:][%][:]
    # other
    | .
''', re.VERBOSE)


STORAGE_CLASS_KEYWORDS = frozenset({
    'typedef', 'extern', 'static', '_Thread_local', 'auto', 'register',
})
TYPE_SPEC_KEYWORDS = frozenset({
    'void', 'char', 'short', 'int', 'long', 'float', 'double', 'signed',
    'unsigned', '_Bool', '_Complex',
    '_Atomic', 'struct', 'union', 'enum',
    '__attribute__',
})
TYPE_QUAL_KEYWORDS = frozenset({
    'const', 'restrict', 'volatile', '_Atomic',
    # GCC
    '__inline',
})
FUNC_SPEC_KEYWORDS = frozenset({'inline', '_Noreturn',})
STATEMENT_KEYWORDS = frozenset({
    'break', 'case', 'continue', 'default', 'do', 'else', 'for', 'goto',
    'if', 'return', 'switch', 'while', '_Static_assert',
})
EXPRESSION_KEYWORDS = frozenset({
    '_Alignas', '_Alignof', '_Imaginary', '_Generic', 'sizeof',
})
DECLSPEC_KEYWORDS = (
    STORAGE_CLASS_KEYWORDS | TYPE_SPEC_KEYWORDS | TYPE_QUAL_KEYWORDS
    | FUNC_SPEC_KEYWORDS
)
KEYWORDS = (
    DECLSPEC_KEYWORDS | STATEMENT_KEYWORDS | EXPRESSION_KEYWORDS
)

class TokenType(enum.Enum):
    KEYWORD = 'K'
    NAME = 'n'
    NUMBER = '#'
    STRING = '"'
    OP = '%'
    OTHER = '?'
    ENDMARKER = '$'

class Token(namedtuple('_Tok', 'type string start end line')):
    def __repr__(self):
        return f'<{self.type.value}{self.string!r}>'


def add_entries(entries, variant=None):
    gcc_proc = subprocess.Popen(
        [
            'gcc',
            '-E', '-dD',
            '-I.', '-IInclude/',
            *shlex.split(CFLAGS),
            *variant.cflags(),
            'Include/Python.h',
        ],
        stdout=subprocess.PIPE,
        encoding='utf-8',
        cwd=projectpath,
        errors='backslashreplace',
    )

    def add_macros_and_yield_tokens():
        def get_entry(name):
            try:
                return entries[name]
            except KeyError:
                entry = entries[name] = Entry(name, [])
                return entry

        def add_definition(definition):
            location = Location(
                filename=filename,
                lineno=lineno,
            )
            entry = get_entry(definition.name)
            # Find the oldest definition that
            # - has the same content (__eq__)
            # - doesn't include this variant yet
            # - isn't followed by another definition of this variant
            #   (e.g. an #undef)
            # If there is one, only add a new variant/location to it.
            # Otherwise, append this `definition` to our entry.
            definition_to_join = None
            for old_definition in reversed(entry.definitions):
                if variant in old_definition.variants:
                    break
                if old_definition == definition:
                    definition_to_join = old_definition
            if definition_to_join:
                definition_to_join.variants[variant] = location
            else:
                definition.variants[variant] = location
                entry.definitions.append(definition)

        def tokenize_line(line):
            for token_match in C_TOKEN_RE.finditer(line):
                token_text = token_match[0]
                toktype = None
                if token_text[0].isspace():
                    continue
                elif token_text[0] in string.ascii_letters + '_':
                    if token_text in KEYWORDS:
                        toktype = TokenType.KEYWORD
                    else:
                        toktype = TokenType.NAME
                elif token_text[0] in string.digits:
                    toktype = TokenType.NUMBER
                elif token_text[0] == '.':
                    toktype = TokenType.NUMBER
                elif token_text[0] in '\'\"':
                    toktype = TokenType.STRING
                else:
                    toktype = TokenType.OP
                tok = Token(
                    toktype,
                    token_text,
                    (lineno, token_match.start(), filename),
                    (lineno, token_match.end(), filename),
                    line,
                )
                tok.filename = filename
                yield tok

        def split_macro_def(content):
            match = re.fullmatch(
                r'([a-zA-Z0-9_-]*)(\([^)]*\))?(.*)',
                content,
            )
            name = match[1]
            args = match[2]
            body = match[3].strip()
            return name, args, body

        filename = '<start>'
        lineno = 0
        preproc_flags = set()
        with gcc_proc.stdout as gcc_stdout:
            for line in gcc_stdout:
                line = line.rstrip('\n')
                try:
                    if line.startswith('# '):
                        preproc_flags = set()
                        rest = line
                        while rest.endswith((' 0', ' 1', ' 2', ' 3', ' 4')):
                            preproc_flags.add(int(rest[-1]))
                            rest = rest[:-2]
                        _hash, lineno, filename = rest.rsplit(None, 2)
                        filename = ast.literal_eval(filename)
                        lineno = int(lineno)
                        continue
                    if line.startswith('#'):
                        if 3 not in preproc_flags and filename != '<built-in>':
                            #print(filename, lineno, line)
                            if line.startswith('#define'):
                                content = line[len('#define'):].strip()
                                name, args, body = split_macro_def(content)
                                add_definition(MacroDef(
                                    kind = 'define',
                                    name=name, args=args, body=body,
                                ))
                            elif line.startswith('#undef'):
                                content = line[len('#undef'):].strip()
                                name, args, body = split_macro_def(content)
                                assert not body
                                add_definition(MacroUndef(
                                    kind='undef',
                                    name=name, args=args,
                                ))
                            else:
                                print(line)
                    else:
                        #if 3 not in preproc_flags and filename != '<builtin>' and line.strip():
                            yield from tokenize_line(line)
                except:
                    print('Line was:', repr(line), f'({filename}:{lineno})')
                    raise
                lineno += 1

    tree = Parser(add_macros_and_yield_tokens()).parse_block()

    if gcc_proc.wait():
        raise Exception(f'gcc failed with return code {gcc_proc.returncode}')

    return tree


@dataclass
class TypedefName:
    name: str
    definition: object = None


@dataclass
class Attribute:
    content: object


@dataclass
class Alignas:
    content: object


@dataclass
class Pointer:
    qualifiers: object
    inner: object

    @cached_property
    def identifier(self):
        return self.inner.identifier


@dataclass
class Declaration:
    storage_classes: list
    type_specs: list
    type_quals: list
    func_specs: list
    alignment_specs: list
    declarators: list

@dataclass
class Declarator:
    content: object

    @cached_property
    def identifier(self):
        match self.content:
            case Declaration(ident):
                return ident
            case None:
                return None
            case _ as ident_token:
                return ident_token.string


@dataclass
class Func:
    declaration: Declaration
    params: object

    @cached_property
    def identifier(self):
        return self.declaration.identifier

@dataclass
class StructType:
    tag: str
    contents: object

@dataclass
class UnionType:
    tag: str
    contents: object

@dataclass
class EnumType:
    tag: str
    contents: object

@dataclass
class If:
    condition: object
    body: object
    orelse: object

@dataclass
class BinExpr:
    left: object
    op: object
    right: object

@dataclass
class Stuff:
    contents: list

    @cached_property
    def string(self):
        return self.opening + ' '.join(x.string for x in self.contents) + self.closing

    def __str__(self):
        return self.string

    def __repr__(self):
        return f'{type(self).__name__}<{self}>'

class ParenthesizedStuff(Stuff):
    opening = '('
    closing = ')'

class BracketedStuff(Stuff):
    opening = '['
    closing = ']'

class BracedStuff(Stuff):
    opening = '{'
    closing = '}'

class SemicolonStuff(Stuff):
    opening = ''
    closing = ';'

Stuff.DELIMITED_CLASSES = ParenthesizedStuff, BracketedStuff, BracedStuff

class Parser:
    def __init__(self, tokens, underscore_t_is_types=False):
        self._underscore_t_is_types = underscore_t_is_types
        self._tokens = iter(tokens)
        self._current_token = None
        self._advance()
        self.main_namespace = ChainMap({})

    def _advance(self):
        prev = self._current_token
        #print(prev); import traceback; traceback.print_stack()
        try:
            self._current_token = next(self._tokens)
        except StopIteration:
            self._current_token = Token(
                TokenType.ENDMARKER, "", (0, 0, ""), (0, 0, ""), ""
            )
        #return
        if (
            prev is None
            or prev.start[0] != self._current_token.start[0]
            or prev.start[-1] != self._current_token.start[-1]
        ):
            print(
                'L', self._current_token.line,
                f'//{self._current_token.filename}:{self._current_token.start[0]}',
            )
        #return
        print('A', self._current_token)

    def peek(self, *want):
        tok = self._current_token
        if self._match(tok, *want):
            return tok
        return None

    def accept(self, *want):
        tok = self._current_token
        if self._match(tok, *want):
            self._advance()
            return tok
        return None

    def expect(self, *want):
        tok = self._current_token
        if not self._match(tok, *want):
            self.throw(f'expected {self._format_want(want)}')
        self._advance()
        return tok

    def _format_want(self, want):
        parts = []
        for w in want:
            match w:
                case str():
                    parts.append(repr(w))
                case TokenType():
                    parts.append(w.name)
        return ' or '.join(parts)

    def _match(self, tok, *want):
        for item in want:
            if item is None or tok.type == item or tok.string == item:
                return True
        #print(tok, '!=~', self._format_want(want))
        return False

    def throw(self, msg=None, bad_token=None):
        if bad_token is None:
            bad_token = self._current_token
        if msg is None:
            msg = f'unexpected {bad_token.string!r}'
        raise SyntaxError(
            msg,
            (
                bad_token.filename,
                bad_token.start[0],
                1 + bad_token.start[1],
                bad_token.line,
            )
        )

    @contextmanager
    def scope(self):
        old = self.main_namespace
        self.main_namespace = old.new_child()
        try:
            yield
        finally:
            self.main_namespace = old

    _depth = 0
    def parsefunc(func):
        def decorated(self, *args, **kwargs):
            depth = self._depth
            pargs = ', '.join([
                *(f'{a!r}' for a in args),
                *(f'{k!r}={v!r}' for k, v in kwargs.items()),
            ])
            print(f'{' ' * depth}{func.__name__}({pargs})@{self._current_token}')
            self._depth += 1
            try:
                result = func(self, *args, **kwargs)
            except BaseException as e:
                result = repr(e)
                raise
            finally:
                self._depth = depth
                print(f'{' ' * depth}{func.__name__} -> {result}')
            return result
        return decorated

    @parsefunc
    def parse_stuff(self, cls):
        contents = []
        result = cls(contents)
        if cls.opening:
            opening_token = self.expect(cls.opening)
        while True:
            if self.accept(cls.closing):
                return result
            if self.peek(TokenType.ENDMARKER):
                if cls.opening:
                    self.throw(
                        f'unclosed {opening_token.string!r} (at end of file)',
                        bad_token=opening_token,
                    )
                else:
                    return result
            for other in Stuff.DELIMITED_CLASSES:
                if self.peek(other.opening):
                    contents.append(self.parse_stuff(other))
                    break
                if self.peek(other.closing):
                    if cls.opening:
                        try:
                            self.throw(
                                f'unclosed {opening_token.string!r}',
                                bad_token=opening_token,
                            )
                        except SyntaxError:
                            self.throw()
                    else:
                        self.throw()
            else:
                contents.append(self.expect(None))


    @parsefunc
    def parse_block(self, end=TokenType.ENDMARKER):
        # In C there should be at least one external declaration. But allowing
        # an empty file is a common extension even in real compilers.
        # Let's do that.
        # Also: we'll allow statements at the top level, and func definitions
        # inside functions. Correct code shouldn't do that.
        result = []
        with self.scope():
            while not self.accept(end):
                item = self.parse_block_item()
                result.append(item)
                if isinstance(item, Declaration) and 'typedef' in item.storage_classes:
                    for declarator in item.declarators:
                        ident = declarator.identifier
                        self.main_namespace[ident] = TypedefName(ident, item)
                #print(';;'.join(self.main_namespace))
        return result

    @parsefunc
    def parse_block_item(self):
        # STATEMENTS
        # - Labeled stmt: identifier ':', 'case', 'default'
        # - Compount stmt: '{'
        # - Expression stmt: anything else
        # - Null stmt: ';'
        # - Selecion stmt: 'if', 'switch'
        # - Iteration stmt: 'while', 'do', 'for'
        # - Jump: 'goto', 'continue', 'break', 'return'
        # DECLARATIONS
        # - declspec keyword
        # - identifier (typedef name)
        if definition := self.peek_typedef_name():
            # TODO: labeled statements
            return self.parse_declaration()
        elif self.peek(TokenType.NAME):
            # TODO: labeled statements
            return self.parse_stuff(SemicolonStuff)
        elif self.peek(*DECLSPEC_KEYWORDS):
            return self.parse_declaration()
        elif self.accept('{'):
            with self.scope():
                return self.parse_block(end='}')
        elif self.peek('if'):
            return self.parse_if()
        elif self.peek(*STATEMENT_KEYWORDS):
            return self.parse_stuff(SemicolonStuff)
        else:
            return self.parse_stuff(SemicolonStuff)

    def peek_typedef_name(self):
        if (
            (tok := self.peek(TokenType.NAME))
            and (definition := self.main_namespace.get(tok.string))
            and isinstance(definition, TypedefName)
        ):
            return definition

    @parsefunc
    def parse_declaration(self):
        result = Declaration(
            storage_classes=[],
            type_specs=[],
            type_quals=[],
            func_specs=[],
            alignment_specs=[],
            declarators=[],
        )
        while True:
            if tok := self.accept(*STORAGE_CLASS_KEYWORDS):
                # There may be at most one of these -- with one exception, so
                # let's just make this a list.
                # Note that 'typedef' is grouped here for convenience.
                result.storage_classes.append(tok.string)
            elif tok := self.accept('struct', 'union', 'enum'):
                tag = None
                decllist = None
                if tagtok := self.accept(TokenType.NAME):
                    tag = tagtok.string
                if self.peek('{'):
                    # will be different for struct/union and enum
                    decllist = self.parse_stuff(BracedStuff)
                if tok.string == 'struct':
                    result.type_specs.append(StructType(tag, decllist))
                elif tok.string == 'union':
                    result.type_specs.append(UnionType(tag, decllist))
                elif tok.string == 'enum':
                    result.type_specs.append(EnumType(tag, decllist))
            elif definition := self.peek_typedef_name():
                result.type_specs.append(definition)
                self.expect(None)
            elif tok := self.peek(TokenType.NAME):
                if (
                    self._underscore_t_is_types
                    and tok.string.endswith('_t')
                    and not any(isinstance(ts, TypedefName)
                                for ts in result.type_specs)
                ):
                    self.main_namespace.maps[-1][tok.string] = TypedefName(
                        tok.string, '<sys>',
                    )
                else:
                    break
            elif tok := self.accept('__attribute__'):
                result.type_quals.append(Attribute(
                    self.parse_stuff(ParenthesizedStuff)
                ))
            elif tok := self.accept('_Atomic'):
                # '_Atomic(typename)' is a specifier;
                # bare '_Atomic' is a qualifier.
                if self.peek('('):
                    result.type_specs.append(
                        Atomic(self.parse_stuff(ParenthesizedStuff))
                    )
                else:
                    result.type_quals.append(tok.string)
            elif tok := self.accept(*TYPE_SPEC_KEYWORDS):
                result.type_specs.append(tok.string)
            elif tok := self.accept(*TYPE_QUAL_KEYWORDS):
                result.type_quals.append(tok.string)
            elif tok := self.accept(*FUNC_SPEC_KEYWORDS):
                result.func_specs.append(tok.string)
            elif tok := self.accept('_Alignas'):
                result.alignment_specs.append(
                    Alignas(self.parse_stuff(ParenthesizedStuff))
                )
            else:
                break
        while True:
            if self.accept('{'):
                result.body = self.parse_block(end='}')
                return result
            elif self.accept(';'):
                return result
            elif self.accept('='):
                result.declarators[-1].init = self.parse_initializer()
            elif self.accept(','):
                pass
            elif self.accept('__attribute__'):
                result.type_quals.append(Attribute(
                    self.parse_stuff(ParenthesizedStuff)
                ))
            else:
                result.declarators.append(self.parse_declarator());

    @parsefunc
    def parse_declarator(self, can_be_abstract=False):
        if self.accept('*'):
            qualifiers = []
            while tok := self.accept(TYPE_QUAL_KEYWORDS):
                qualifiers.append(tok.string)
            return Pointer(
                qualifiers,
                self.parse_declarator(),
            )
        else:
            return self.parse_direct_declarator()

    @parsefunc
    def parse_direct_declarator(self, can_be_abstract=False):
        if self.accept('('):
            result = self.parse_declarator()
            self.expect(')')
        elif tok := self.accept(TokenType.NAME):
            result = Declarator(tok)
        elif can_be_abstract:
            result = Declarator(None)
        else:
            self.throw()
        while tok := self.peek('[', '('):
            if tok.string == '[':
                result = Array(result, self.parse_stuff(BracketedStuff))
            else:
                result = Func(result, self.parse_stuff(ParenthesizedStuff))
                result.attrs = []
                while self.accept('__attribute__'):
                    result.attrs.append(Attribute(
                        self.parse_stuff(ParenthesizedStuff)
                    ))
        return result

    @parsefunc
    def parse_if(self):
        self.expect('if')
        self.expect('(')
        cond = self.parse_expression()
        self.expect(')')
        stmt = If(cond, self.parse_block_item(), None)
        if self.accept('else'):
            stmt.orelse = self.parse_block_item()
        return stmt

    @parsefunc
    def parse_initializer(self):
        if self.peek('{'):
            return self.parse_stuff(BracedStuff)
        return self.parse_assignment_expression()

    @parsefunc
    def parse_expression(self):
        exp = self.parse_assignment_expression()
        if tok := self.accept(','):
            return BinExpr(
                exp,
                tok.string,
                self.parse_expression(),
            )
        return exp

    @parsefunc
    def parse_assignment_expression(self):
        exp = self.parse_conditional_expression()
        if tok := self.accept(*'= *= /= %= += -= <<= >>= &= ^= |='.split()):
            return BinExpr(
                exp,
                tok.string,
                self.parse_assignment_expression(),
            )
        return exp

    @parsefunc
    def parse_conditional_expression(self):
        exp = self.parse_logor_expression()
        if tok := self.accept('?'):
            truthy = self.parse_expression()
            self.expect(':')
            falsy = self.parse_conditional_expression()
            return TerExpr(exp, truthy, falsy)
        return exp

    @parsefunc
    def parse_logor_expression(self):
        exp = self.parse_logand_expression()
        if tok := self.accept('||'):
            return BinExpr(
                exp,
                tok.string,
                self.parse_logor_expression()
            )
        return exp

    @parsefunc
    def parse_logand_expression(self):
        exp = self.parse_or_expression()
        if tok := self.accept('&&'):
            return BinExpr(
                exp,
                tok.string,
                self.parse_logand_expression()
            )
        return exp

    @parsefunc
    def parse_or_expression(self):
        exp = self.parse_xor_expression()
        if tok := self.accept('|'):
            return BinExpr(
                exp,
                tok.string,
                self.parse_or_expression()
            )
        return exp

    @parsefunc
    def parse_xor_expression(self):
        exp = self.parse_and_expression()
        if tok := self.accept('^'):
            return BinExpr(
                exp,
                tok.string,
                self.parse_xor_expression()
            )
        return exp

    @parsefunc
    def parse_and_expression(self):
        exp = self.parse_eq_expression()
        if tok := self.accept('&'):
            return BinExpr(
                exp,
                tok.string,
                self.parse_and_expression()
            )
        return exp

    @parsefunc
    def parse_eq_expression(self):
        exp = self.parse_rel_expression()
        if tok := self.accept('==', '!='):
            return BinExpr(
                exp,
                tok.string,
                self.parse_eq_expression()
            )
        return exp

    @parsefunc
    def parse_rel_expression(self):
        exp = self.parse_shift_expression()
        if tok := self.accept('<', '>', '<=', '>='):
            return BinExpr(
                exp,
                tok.string,
                self.parse_rel_expression()
            )
        return exp

    @parsefunc
    def parse_shift_expression(self):
        exp = self.parse_additive_expression()
        if tok := self.accept('<<', '>>'):
            return BinExpr(
                exp,
                tok.string,
                self.parse_shift_expression()
            )
        return exp

    @parsefunc
    def parse_additive_expression(self):
        exp = self.parse_multiplicative_expression()
        if tok := self.accept('+', '-'):
            return BinExpr(
                exp,
                tok.string,
                self.parse_additive_expression()
            )
        return exp

    @parsefunc
    def parse_multiplicative_expression(self):
        exp = self.parse_cast_expression()
        if tok := self.accept('*', '/', '%'):
            return BinExpr(
                exp,
                tok.string,
                self.parse_multiplicative_expression()
            )
        return exp

    @parsefunc
    def parse_cast_expression(self):
        # XXX
        return self.parse_unary_expression()

    @parsefunc
    def parse_unary_expression(self, or_type=False):
        if tok := self.accept('&', '*', '+', '-', '~', '!'):
            return UnExpr(tok.string, self.parse_cast_expression())
        if tok := self.accept('++', '--'):
            return UnExpr('pre' + tok.string, self.parse_unary_expression())
        if tok := self.accept('sizeof'):
            return UnExpr(tok.string, self.parse_unary_expression(True))
        if tok := self.accept('_Alignof'):
            return UnExpr(tok.string, self.parse_stuff(ParenthesizedStuff))
        return self.parse_postfix_expression(or_type)

    @parsefunc
    def parse_postfix_expression(self, or_type=False):
        exp = self.parse_primary_expression()
        while self.peek('[', '(', '.', '->', '++', '--'):
            if self.accept('['):
                exp = BinExpr(exp, '[]', self.parse_expression())
                self.expect(']')
            elif self.peek('('):
                exp = BinExpr(exp, '()', self.parse_stuff(ParenthesizedStuff))
            elif tok := self.peek('.', '->'):
                exp = BinExpr(exp, tok.string, self.expect(TokenType.NAME))
            elif tok := self.peek('++', '--'):
                exp = UnExpr('post' + tok.string, exp)
        return exp

    @parsefunc
    def parse_primary_expression(self):
        if tok := self.accept(
            TokenType.NAME, TokenType.NUMBER, TokenType.STRING,
        ):
            return tok
        elif tok := self.accept('('):
            expr = self.parse_expression()
            self.expect(')')
            return expr
        else:
            self.expect('_Generic')
            return self.parse_stuff(ParenthesizedStuff)

entries = {}
# add_entries(entries, Variant(None))
tree = add_entries(entries, Variant(3))
# add_entries(entries, Variant(0x03020000))
# add_entries(entries, Variant(0x03030000))
# add_entries(entries, Variant(0x03040000))
# add_entries(entries, Variant(0x03050000))
# add_entries(entries, Variant(0x03060000))
# add_entries(entries, Variant(0x03070000))
# add_entries(entries, Variant(0x03080000))
# add_entries(entries, Variant(0x03090000))
# add_entries(entries, Variant(0x030a0000))
# add_entries(entries, Variant(0x030b0000))
# add_entries(entries, Variant(0x030c0000))
# add_entries(entries, Variant(0x030d0000))

@lru_cache
def format_variant_range(variants):
    variants = sorted(variants)
    results = []
    for variant in variants:
        if results and results[-1][-1].next == variant:
            results[-1][-1] = variant
        else:
            results.append([variant, variant])
    return '[' + ', '.join(
        'L' if start == Variant(3) and stop == Variant(0x030d0000)
        else f'{start}+' if stop == Variant(0x030d0000)
        else f'{start}' if start == stop
        else f'{start}-{stop}'
        for start, stop in results
    ) + ']'

for entry in entries.values():
    color = CYAN
    if not entry.name.startswith(('Py', 'PY', '_Py', '_PY')):
        color = RED
    print(f'{color}{entry.name}{RESET}')
    for definition in entry.definitions:
        loc_to_var = {}
        for variant, location in definition.variants.items():
            loc_to_var.setdefault(location, []).append(variant)
        print(' ' * 3, definition.colorized)
        for location, variants in loc_to_var.items():
            print(' ' * 7, format_variant_range(frozenset(variants)), location)

print(tree)
