from functools import lru_cache, total_ordering, cached_property
from dataclasses import dataclass, field
from collections import namedtuple
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

class TokenType(enum.Enum):
    NAME = 'n'
    NUMBER = '#'
    STRING = '"'
    OP = '%'
    OTHER = '?'
    ENDMARKER = '$'

class Token(namedtuple('_Tok', 'type string start end line')):
    def __repr__(self):
        return f'<{self.string!r}>'


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
        # - isn't followed by another definition of this variant (e.g. #undef)
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
            elif token_text[0] in string.ascii_letters:
                toktype = TokenType.NAME
            elif token_text[0] == '_':
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
    c_tokens = []
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
                    if 3 not in preproc_flags and filename != '<builtin>' and line.strip():
                        c_tokens.extend(tokenize_line(line))
            except:
                print('Line was:', repr(line), f'({filename}:{lineno})')
                raise
            lineno += 1

    tree = Parser(c_tokens).parse_translation_unit()

    if gcc_proc.wait():
        raise Exception(f'gcc failed with return code {gcc_proc.returncode}')

    return tree


@dataclass
class TypedefName:
    name: str
    definition: object


class Parser:
    def __init__(self, tokens):
        self._tokens = iter(tokens)
        self._current_token = None
        self._advance()
        self.main_namespace = {
            'uintptr_t': TypedefName('uintptr_t', 'unsigned int*'),
        }

    def _advance(self):
        prev = self._current_token
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
            print('L', self._current_token.line)
        #return
        print('A', self._current_token)

    def peek(self, *want):
        tok = self._current_token
        if self._match(tok, *want):
            return tok
        return None

    def try_eat(self, *want):
        tok = self._current_token
        if self._match(tok, *want):
            self._advance()
            return tok
        return None

    def eat(self, *want):
        tok = self._current_token
        if not self._match(tok, *want):
            self.throw(f'expected {self._format_want(want)}')
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
            if want is None or tok.type == item or tok.string == item:
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

    _depth = 0
    def parsefunc(func):
        def decorated(self, *args, **kwargs):
            depth = self._depth
            print(f'{' ' * depth}{func.__name__}(*{args}, **{kwargs})@{self._current_token}')
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
    def parse_translation_unit(self):
        #   | external-declaration+
        result = []
        while True:
            result.append(self.parse_external_declaration())
            if self.peek(TokenType.ENDMARKER):
                break
        return result

    @parsefunc
    def parse_external_declaration(self):
        # external_declaration:
        #   | function_definition
        #   | declaration
        # function_definition:
        #   | declaration_specifiers declarator [declaration+] compound_statement
        #       (but we omit the old-style [declaration+])
        # declaration:
        #   | declaration_specifiers [','.init-declarator+] ';'
        #   | static_assert_declaration
        # static_assert_declaration:
        #   | '_Static_assert' '(' constant-expression ',' string-literal ')' ';'
        if self.peek('_Static_assert'):
            return self.parse_static_assert_declaration()
        declaration_specifiers = self.parse_declaration_specifiers()
        declarator = self.parse_init_declarator()
        if self.peek('{'):
            body = self.parse_compound_statement()
            return ['FD', declaration_specifiers, declarator, body]
        else:
            declarators = [declarator]
            while self.peek(','):
                declarators.append(self.parse_init_declarator())
            self.eat(';')
            return ['ED', declaration_specifiers, declarators]

    @parsefunc
    def parse_function_definition(self):
        return self.parse_any(one=True)

    @parsefunc
    def parse_init_declarator(self):
        declarator = self.parse_declarator()
        if self.try_eat('='):
            init = self.parse_initializer()
            return [declarator, init]
        return declarator

    @parsefunc
    def parse_static_assert_declaration(self):
        # static_assert_declaration:
        #   | '_Static_assert' '(' constant-expression ',' string-literal ')' ';'
        return self.parse_any(one=True)

    @parsefunc
    def parse_declaration_specifiers(self):
        # declaration_specifiers:
        #   | storage-class-specifier [declaration-specifiers]
        #   | type-specifier [declaration-specifiers]
        #   | type-qualifier [declaration-specifiers
        #   | function-specifier [declaration-specifiers]
        #   | alignment-specifier [declaration-specifiers]
        specifiers = []
        while True:
            # storage-class-specifier
            if tok := self.try_eat(
                'typedef', 'extern', 'static', '_Thread_local',
                'auto', 'register'
            ):
                specifiers.append(tok.string)
            # type-specifier
            elif tok := self.try_eat(
                'void', 'char', 'short', 'int', 'long', 'float', 'double',
                'signed', 'unsigned', '_Bool', '_Complex'
                # atomic; struct-or-union; enum; typedef-name
            ):
                specifiers.append(tok.string)
            elif tok := self.try_eat('_Atomic'):
                self.expect('(')
                specifiers.append((tok.string, self.parse_any()))
            elif tok := self.try_eat('struct', 'union', 'enum'):
                ident = self.try_eat(TokenType.STRING)
                if self.peek('{'):
                    if toke.string == 'enum':
                        sdl = self.parse_enumerator_list()
                    else:
                        sdl = self.parse_struct_declaration_list()
                else:
                    sdl = None
                if ident is None and sdl is None:
                    self.throw()
                specifiers.append([tok.string, ident, sdl])
            elif (
                (tok := self.peek(TokenType.NAME))
                and isinstance(
                    (td := self.main_namespace.get(tok.string)),
                    TypedefName,
                )
            ):
                self.eat(TokenType.NAME)
                specifiers.append(td)
            # type-qualifier
            elif tok := self.try_eat(
                'const', 'restrict', 'volatile', '_Atomic',
            ):
                specifiers.append(tok.string)
            # function-specifier
            elif tok := self.try_eat(
                'inline', '_Noreturn',
            ):
                specifiers.append(tok.string)
            # alignment-specifier
            #   | '_Alignas' ( type-name )
            #   | '_Alignas' ( constant-expression )
            elif tok := self.try_eat('_Alignas'):
                self.expect('(')
                specifiers.append((tok.string, self.parse_any()))
            else:
                if not specifiers:
                    self.throw()
                break
        return specifiers

    @parsefunc
    def parse_declarator(self):
        if self.try_eat('*'):
            qualifiers = []
            while tok := self.try_eat(
                'const', 'restrict', 'volatile', '_Atomic',
            ):
                qualifiers.append(tok.string)
            return ['*', qualifiers, self.parse_declarator()]
        return self.parse_direct_declarator()

    @parsefunc
    def parse_direct_declarator(self):
        if tok := self.try_eat(TokenType.NAME):
            return tok.string
        if self.peek('('):
            return self.parse_any()
        while self.peek('(', '['):
            raise TODO


    @parsefunc
    def parse_struct_declaration_list(self):
        return self.parse_any(one=True)

    @parsefunc
    def parse_enumerator_list(self):
        return self.parse_any(one=True)

    @parsefunc
    def parse_any(self, one=False):
        result = []
        while True:
            cont = False
            for opening, closing in '()', '[]', '{}':
                if self.try_eat(opening):
                    result.append([opening, *parse_any(self), closing])
                    if not self.try_eat(closing):
                        self._raise(f'unclosed {closing!r}', bad_token=tok)
                    break
                if self.peek(closing):
                    break
            else:
                if self.peek(TokenType.ENDMARKER):
                    break
                if one and (semi := self.try_eat(';')):
                    result.append(semi)
                    break
                result.append(self.eat(None))
        return result

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
