from functools import lru_cache, total_ordering, cached_property
from dataclasses import dataclass, field
from collections import namedtuple
from pathlib import Path
import subprocess
import sysconfig
import string
import shlex
import token
import ast
import sys
import re

try:
    import pegen
except ModuleNotFoundError:
    sys.path.append(str(Path(__file__).parent.parent.resolve() / 'peg_generator'))
    import pegen

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

class Token(namedtuple('_Tok', 'type string start end line')):
    pass


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
                toktype = token.NAME
            elif token_text[0] == '_':
                toktype = token.NAME
            elif token_text[0] in string.digits:
                toktype = token.NUMBER
            elif token_text[0] == '.':
                toktype = token.NUMBER
            elif token_text[0] in '\'\"':
                toktype = token.STRING
            else:
                toktype = token.OP
            tok = Token(
                toktype,
                token_text,
                (lineno, token_match.start()),
                (lineno, token_match.end()),
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

    c_tokens.append(Token(token.ENDMARKER, "", (0, 0), (0, 0), ""))

    import cparser
    from pegen.tokenizer import Tokenizer
    tokenizer = Tokenizer(iter(c_tokens), verbose=False)
    parser = cparser.GeneratedParser(tokenizer, verbose=True)
    tree = parser.start()
    if not tree:
        raise parser.make_syntax_error('Include.h')

    if gcc_proc.wait():
        raise Exception(f'gcc failed with return code {gcc_proc.returncode}')

    return tree

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
