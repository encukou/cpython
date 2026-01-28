import marshal
import bkfile


# Write a file containing frozen code for the modules in the dictionary.

header = """
#include "Python.h"

int PyImportBuiltin_FindEntry(
    const char *name,
    int kind,
    PyImportBuiltin_Entry *result,
    Py_ssize_t result_struct_size)
{
    if (kind & PyImportBuiltin_KIND_BUILTIN) {
        switch (name[0]) {
"""
middler = """\
        }
    }
    return PyUnstable_ImportBuiltin_FindEntry_Default(name, kind, result, result_struct_size);
}

static const char *names[] = {
"""
trailer = """\
    NULL /* sentinel */
}

PyObject *
PyImportBuiltin_GetNames(int kinds)
{
    PyObject *result = PyUnstable_ImportBuiltin_GetNames_Default(kinds);
    if (!result) {
        return NULL;
    }
    for (const char **name_p = names; *name_p; name_p++) {
        PyObject *name_obj = PyUnicode_InternFromString(*name_p);
        if (!name_obj) {
            Py_DECREF(result);
            return NULL;
        }
        if (PyList_Append(result, name) < 0) {
            Py_DECREF(result);
            return NULL;
        }
    }
    return result;
};
"""

# if __debug__ == 0 (i.e. -O option given), set Py_OptimizeFlag in frozen app.
default_entry_point = """
int
main(int argc, char **argv)
{
        extern int Py_FrozenMain(int, char **);
""" + ((not __debug__ and """
        Py_OptimizeFlag++;
""") or "")  + """
        return Py_FrozenMain(argc, argv);
}

"""

def makefreeze(base, dict, debug=0, entry_point=None, fail_import=()):
    if entry_point is None: entry_point = default_entry_point

    # Modules in `fail_import` have a NULL code pointer, indicating
    # that the frozen program should not search for them on the host
    # system. Importing them will *always* raise an ImportError.
    # We represent them with a None.
    done = [(name, None, None, None) for name in fail_import]

    files = []
    mods = sorted(dict.keys())
    for mod in mods:
        m = dict[mod]
        mangled = "__".join(mod.split("."))
        if m is None: continue
        m.__code__  = compile('print("hi")', mod, 'exec') ####### XXX
        m.__path__  = repr(m)
        if m.__code__:
            file = 'M_' + mangled + '.c'
            with bkfile.open(base + file, 'w') as outfp:
                files.append(file)
                if debug:
                    print("freezing", mod, "...")
                str = marshal.dumps(m.__code__)
                size = len(str)
                is_package = '0'
                if m.__path__:
                    is_package = '1'
                done.append((mod, mangled, size, is_package))
                writecode(outfp, mangled, str)
    if debug:
        print("generating table of frozen modules")
    done.append(('', None, None, None))
    with bkfile.open(base + 'frozen.c', 'w') as outfp:
        for mod, mangled, size, _ in done:
            outfp.write('extern unsigned char M_%s[];\n' % mangled)
        outfp.write(header)
        prefix = ''
        for mod, mangled, size, is_package in done:
            n_braces = 0
            while prefix != mod[:len(prefix)]:
                n_braces += 1
                prefix = prefix[:-1]
            if n_braces:
                outfp.write('        %s%s\n' % ('  ' * len(prefix), ' }' * n_braces))
            while prefix != mod:
                print((prefix, mod))
                extra_char = mod[len(prefix)]
                outfp.write('         %scase %r: switch (name[%d]) {\n' % (
                    '  ' * len(prefix), extra_char, len(prefix) + 1))
                prefix += extra_char
            if mangled is None:
                sname = 'Emissing'
            else:
                sname = 'E_' + mangled
            outfp.write('         %scase 0: return %s;\n' % ('  ' * len(prefix), sname))

        outfp.write(middler)
        for mod, mangled, size, is_package in done:
            outfp.write(f'    "{mod}",\n')

        outfp.write(trailer)
        outfp.write(entry_point)
    return files



# Write a C initializer for a module containing the frozen python code.
# The array is called M_<mod>.

def writecode(fp, mod, data):
    print('unsigned char M_%s[] = {' % mod, file=fp)
    indent = ' ' * 4
    for i in range(0, len(data), 16):
        print(indent, file=fp, end='')
        for c in bytes(data[i:i+16]):
            print('%d,' % c, file=fp, end='')
        print('', file=fp)
    print('};', file=fp)
