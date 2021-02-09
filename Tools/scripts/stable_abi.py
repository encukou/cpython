"""Check the stable ABI manifest or generate files from it

By default, the tool only checks existing files.
Pass --generate to recreate auto-generated files instead.

For actions that take a FILENAME, the filename can be left out to use a default
(relative to the manifest file, as they appear in the CPython codebase).
"""
from functools import partial
from pathlib import Path
import dataclasses
import subprocess
import sysconfig
import argparse
import textwrap
import difflib
import shutil
import sys
import os
import io
import re

MISSING = object()

EXCLUDED_HEADERS = {
    "bytes_methods.h",
    "cellobject.h",
    "classobject.h",
    "code.h",
    "compile.h",
    "datetime.h",
    "dtoa.h",
    "frameobject.h",
    "funcobject.h",
    "genobject.h",
    "longintrepr.h",
    "parsetok.h",
    "pyarena.h",
    "pyatomic.h",
    "pyctype.h",
    "pydebug.h",
    "pytime.h",
    "symtable.h",
    "token.h",
    "ucnhash.h",
}
MACOS = (sys.platform == "darwin")
UNIXY = MACOS or (sys.platform == "linux")


@dataclasses.dataclass
class Manifest:
    kind = 'manifest'
    contents: dict = dataclasses.field(default_factory=dict)

    def add(self, item):
        if item.name in self.contents:
            raise ValueError(f'duplicate ABI item {item.name}')
        self.contents[item.name] = item

    @property
    def feature_macros(self):
        """Macros like HAVE_FORK and MS_WINDOWS which affect what's available"""
        return set(item.ifdef for item in self.contents.values()) - {None}

    def select(self, kinds, *, include_abi_only=True, ifdef=None):
        for name, item in sorted(self.contents.items()):
            if item.kind not in kinds:
                continue
            if item.abi_only and not include_abi_only:
                continue
            if (ifdef is not None
                    and item.ifdef is not None
                    and item.ifdef not in ifdef):
                continue
            yield item

    def dump(self, indent=0):
        print('  ' * indent, 'manifest', sep='')
        for item in self.contents.values():
            item.dump(indent+2)

@dataclasses.dataclass
class ABIItem:
    kind: str
    name: str
    contents: list = dataclasses.field(default_factory=list)
    abi_only: bool = False
    ifdef: str = None

    def append(self, item):
        self.contents.append(item)

    def dump(self, indent=0):
        print('    ' * indent, self.kind, self.name)
        if self.ifdef:
            print('    ' * (indent+1), 'ifdef', self.ifdef)
        if self.abi_only:
            print('    ' * (indent+1), 'abi_only')
        for item in self.contents:
            item.dump(indent+1)

TOPLEVEL = frozenset({
    'function', 'struct', 'macro', 'data', 'const', 'typedef',
})


def parse_manifest(file):
    LINE_RE = re.compile('^(?P<indent>[ ]*)(?P<kind>[^ ]+)[ ]*(?P<content>.*)$')
    manifest = Manifest()
    levels = [(-1, manifest)]
    for lineno, line in enumerate(file, start=1):
        line, sep, comment = line.partition('#')
        line = line.rstrip()
        if not line:
            continue
        match = LINE_RE.match(line)
        if not match:
            raise SyntaxError(f'line {lineno}: invalid syntax: {line}')
        level = len(match['indent'])
        kind = match['kind']
        content = match['content']
        while level <= levels[-1][0]:
            levels.pop()
        parent = levels[-1][1]
        entry = None
        if parent is None:
            pass
        elif kind in TOPLEVEL:
            if parent.kind not in {'manifest'}:
                raise ValueError(f'{kind} cannot go in {parent.kind}')
            entry = ABIItem(kind, content)
            parent.add(entry)
        elif kind in {'hard_removed'}:
            # This item is not available in the API any more. Remove it.
            # XXX: hacky; should remove the items from the manifest instead
            if parent.kind not in TOPLEVEL:
                raise ValueError(f'{kind} cannot go in {parent.kind}')
            removed_entry = manifest.contents.pop(parent.name)
            assert removed_entry == parent
        elif kind in {'ifdef'}:
            if parent.kind not in TOPLEVEL:
                raise ValueError(f'{kind} cannot go in {parent.kind}')
            parent.ifdef = content
        elif kind in {'abi_only'}:
            if parent.kind not in {'function', 'data'}:
                raise ValueError(f'{kind} cannot go in {parent.kind}')
            parent.abi_only = True
        elif kind in {'added', 'field', 'type', 'arg', 'return', 'removed'}:
            # This kind is ignored and unused.
            # XXX: Remove this information from the manifest if it can't be
            # used and/or tested.
            pass
        else:
            raise ValueError(f"unknown kind {kind!r}")
        levels.append((level, entry))
    return manifest


generators = []
def generator(var_name, default_path):
    """Decorates a file generator: function that writes to a file"""
    def _decorator(func):
        func.var_name = var_name
        func.arg_name = '--' + var_name.replace('_', '-')
        func.default_path = default_path
        generators.append(func)
        return func
    return _decorator


@generator("doc_file", 'Doc/data/stable_abi.txt')
def gen_doc(manifest, args, outfile):
    """Generate/check the stable ABI list for documentation"""
    write = partial(print, file=outfile)
    write("# File generated by Tools/stable_abi.py")
    write()
    for item in manifest.select(TOPLEVEL, include_abi_only=False):
        write(item.name)


@generator("python3dll", 'PC/python3dll.c')
def gen_python3dll(manifest, args, outfile):
    """Generate/check the source for the Windows stable ABI library"""
    write = partial(print, file=outfile)
    write(textwrap.dedent(r"""
        /* Re-export stable Python ABI */

        /* Generated by Tools/stable_abi */

        #ifdef _M_IX86
        #define DECORATE "_"
        #else
        #define DECORATE
        #endif

        #define EXPORT_FUNC(name) \
            __pragma(comment(linker, "/EXPORT:" DECORATE #name "=" PYTHON_DLL_NAME "." #name))
        #define EXPORT_DATA(name) \
            __pragma(comment(linker, "/EXPORT:" DECORATE #name "=" PYTHON_DLL_NAME "." #name ",DATA"))
    """), file=outfile)
    for item in manifest.select({'function'}, include_abi_only=True):
        write(f'EXPORT_FUNC({item.name})')
    for item in manifest.select({'data'}, include_abi_only=True):
        write(f'EXPORT_DATA({item.name})')


def generate(manifest, args, path, func):
    if args.generate:
        with path.open('w') as outfile:
            func(manifest, args, outfile)
        return True

    outfile = io.StringIO()
    func(manifest, args, outfile)
    expected = outfile.getvalue()
    got = path.read_text()
    if expected != got:
        print(f'File {path} differs from expected!')
        diff = difflib.unified_diff(
            expected.splitlines(), got.splitlines(),
            str(path), '<expected>',
            lineterm='',
        )
        for line in diff:
            print(line)
    return True


def do_unixy_check(manifest, args):
    """Check headers & library using "Unixy" tools (GCC/clang, binutils)"""
    okay = True

    # Get all macros first: we'll need feature macros like HAVE_FORK and
    # MS_WINDOWS for everything else
    present_macros = gcc_get_limited_api_macros(['Include/Python.h'])
    feature_macros = manifest.feature_macros & present_macros

    # Check that we have all neded macros
    expected_macros = set(
        item.name for item in manifest.select({'macro'})
    )
    missing_macros = expected_macros - present_macros
    if missing_macros:
        print(textwrap.dedent(f"""\
            Some macros from are not defined from "Include/Python.h"
            with Py_LIMITED_API: {', '.join(sorted(missing_macros))}
        """), file=sys.stderr)
        okay = False

    expected_symbols = set(item.name for item in manifest.select(
        {'function', 'data'}, include_abi_only=True, ifdef=feature_macros,
    ))

    # Check the static library (*.a)
    LIBRARY = sysconfig.get_config_var("LIBRARY")
    if not LIBRARY:
        raise Exception("failed to get LIBRARY variable from sysconfig")
    if not binutils_check_library(
            manifest, args, LIBRARY, expected_symbols,
            dynamic=False, ifdef=feature_macros,
            ):
        okay = False

    # Check the dynamic library (*.so)
    LDLIBRARY = sysconfig.get_config_var("LDLIBRARY")
    if not LDLIBRARY:
        raise Exception("failed to get LDLIBRARY variable from sysconfig")
    if not binutils_check_library(
            manifest, args, LDLIBRARY, expected_symbols,
            dynamic=False, ifdef=feature_macros,
            ):
        okay = False

    # Check definitions in the header files
    found_defs = gcc_get_limited_api_definitions(['Include/Python.h'])
    missing_defs = expected_symbols - found_defs
    if missing_macros:
        print(textwrap.dedent(f"""\
            Some expected declarations were not declared in "Include/Python.h"
            with Py_LIMITED_API: {', '.join(sorted(missing_macros))}
        """), file=sys.stderr)
    extra_defs = found_defs - expected_symbols
    if extra_defs:
        print(textwrap.dedent(f"""\
            Some extra declarations were found in "Include/Python.h"
            with Py_LIMITED_API: {', '.join(sorted(extra_defs))}\
        """), file=sys.stderr)
        print("(this error is ignored for now)", file=sys.stderr)
        # XXX: Ideally this should fail the check
        #okay = False

    return okay


def binutils_get_exported_symbols(library, dynamic=False):
    """Retrieve exported symbols using the nm(1) tool from binutils"""
    # Only look at dynamic symbols
    args = ["nm", "--no-sort"]
    if dynamic:
        args.append("--dynamic")
    args.append(library)
    proc = subprocess.run(args, stdout=subprocess.PIPE, universal_newlines=True)
    if proc.returncode:
        sys.stdout.write(proc.stdout)
        sys.exit(proc.returncode)

    stdout = proc.stdout.rstrip()
    if not stdout:
        raise Exception("command output is empty")

    for line in stdout.splitlines():
        # Split line '0000000000001b80 D PyTextIOWrapper_Type'
        if not line:
            continue

        parts = line.split(maxsplit=2)
        if len(parts) < 3:
            continue

        symbol = parts[-1]
        if MACOS and symbol.startswith("_"):
            yield symbol[1:]
        else:
            yield symbol


def binutils_check_library(manifest, args, library, expected_symbols, dynamic, ifdef):
    """Check that library exports all expected_symbols"""
    available_symbols = set(binutils_get_exported_symbols(library, dynamic))
    missing_symbols = expected_symbols - available_symbols
    if missing_symbols:
        print(textwrap.dedent(f"""\
            Some symbols from the limited API are missing from {library}:
                {', '.join(missing_symbols)}

            This error means that there are some missing symbols among the
            ones exported in the library.
            This normally means that some symbol, function implementation or
            a prototype belonging to a symbol in the limited API has been
            deleted or is missing.
        """), file=sys.stderr)
        return False
    return True


def gcc_get_limited_api_macros(headers):
    """Run the preprocesor over all the header files in "Include" setting
    "-DPy_LIMITED_API" to the correct value for the running version of the
    interpreter and extracting all macro definitions (via adding -dM to the
    compiler arguments).

    Requires Python built with a GCC-compatible compiler. (clang might work)
    """

    preprocesor_output_with_macros = subprocess.check_output(
        sysconfig.get_config_var("CC").split()
        + [
            # Prevent the expansion of the exported macros so we can capture them later
            "-DSIZEOF_WCHAR_T=4",  # The actual value is not important
            f"-DPy_LIMITED_API={sys.version_info.major << 24 | sys.version_info.minor << 16}",
            "-I.",
            "-I./Include",
            "-dM",
            "-E",
        ]
        + [str(file) for file in headers],
        text=True,
    )

    return {
        target
        for target in re.findall(
            r"#define (\w+)", preprocesor_output_with_macros
        )
    }


def gcc_get_limited_api_definitions(headers):
    """Run the preprocesor over all the header files in "Include" setting
    "-DPy_LIMITED_API" to the correct value for the running version of the
    interpreter.

    The limited API symbols will be extracted from the output of this command
    as it includes the prototypes and definitions of all the exported symbols
    that are in the limited api.

    This function does *NOT* extract the macros defined on the limited API

    Requires Python built with a GCC-compatible compiler. (clang might work)
    """
    api_hexversion = sys.version_info.major << 24 | sys.version_info.minor << 16
    preprocesor_output = subprocess.check_output(
        sysconfig.get_config_var("CC").split()
        + [
            # Prevent the expansion of the exported macros so we can capture
            # them later
            "-DPyAPI_FUNC=__PyAPI_FUNC",
            "-DPyAPI_DATA=__PyAPI_DATA",
            "-DEXPORT_DATA=__EXPORT_DATA",
            "-D_Py_NO_RETURN=",
            "-DSIZEOF_WCHAR_T=4",  # The actual value is not important
            f"-DPy_LIMITED_API={api_hexversion}",
            "-I.",
            "-I./Include",
            "-E",
        ]
        + [str(file) for file in headers],
        text=True,
        stderr=subprocess.DEVNULL,
    )
    stable_functions = set(
        re.findall(r"__PyAPI_FUNC\(.*?\)\s*(.*?)\s*\(", preprocesor_output)
    )
    stable_exported_data = set(
        re.findall(r"__EXPORT_DATA\((.*?)\)", preprocesor_output)
    )
    stable_data = set(
        re.findall(r"__PyAPI_DATA\(.*?\)\s*\(?(.*?)\)?\s*;", preprocesor_output)
    )
    return stable_data | stable_exported_data | stable_functions


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "file", type=Path, metavar='FILE',
        help="file with the stable abi manifest",
    )
    parser.add_argument(
        "--generate", action='store_true',
        help="generate file(s), rather than just checking them",
    )
    parser.add_argument(
        "-a", "--all", action='store_true',
        help="run all available actions using their default filenames",
    )
    parser.add_argument(
        "-l", "--list", action='store_true',
        help="list all actions and their default filenames; then exit",
    )
    parser.add_argument(
        "--dump", action='store_true',
        help="dump the manifest contents",
    )
    actions_group = parser.add_argument_group('actions')
    for gen in generators:
        actions_group.add_argument(
            gen.arg_name, dest=gen.var_name,
            type=str, nargs="?", default=MISSING,
            metavar='FILENAME',
            help=gen.__doc__,
        )
    actions_group.add_argument(
        '--unixy-check', action='store_true',
        help=do_unixy_check.__doc__,
    )
    args = parser.parse_args()

    base_path = args.file.parent.parent

    if args.list:
        for gen in generators:
            print(f'{gen.arg_name}: {base_path / gen.default_path}')
        sys.exit(0)

    if args.all and UNIXY:
        args.unixy_check = True

    with args.file.open() as file:
        manifest = parse_manifest(file)

    # Remember results of all actions (as booleans).
    # At the end we'll check that all were OK and at least one actions was run.
    results = {}

    if args.dump:
        manifest.dump()
        results.append(True)

    for gen in generators:
        filename = getattr(args, gen.var_name)
        if filename is None or (filename is MISSING and args.all):
            filename = base_path / gen.default_path
        elif filename is MISSING:
            continue

        results[gen.var_name] = generate(manifest, args, filename, gen)

    if args.unixy_check:
        results['unixy_check'] = do_unixy_check(manifest, args)

    if not results:
        if args.generate:
            parser.error('No file specified. Use --help for usage.')
        parser.error('No check specified. Use --help for usage.')

    failed_results = [name for name, result in results.items() if not result]

    if failed_results:
        raise Exception(f"""
        These checks related to the stable ABI did not succeed:
            {', '.join(failed_results)}

        If you see diffs in the output, files derived from the stable
        ABI manifest the were not regenerated.
        Run `make regen-abi` to fix this.

        Otherwise, see the errors above.

        The stable ABI manifest is at: {args.file}
        Note that there is a process to follow when modifying it.

        You can read more about the limited API and its contracts at:

        https://docs.python.org/3/c-api/stable.html

        And in PEP 384:

        https://www.python.org/dev/peps/pep-0384/
        """)


if __name__ == "__main__":
    main()
