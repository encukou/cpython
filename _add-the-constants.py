import subprocess

import tomlkit

# /// script
# requires-python = ">=3.15"
# dependencies = ["tomlkit"]
# ///

toml_content = subprocess.check_output(['git', 'show', 'main:Misc/stable_abi.toml'])
manifest = tomlkit.loads(toml_content)

print(manifest)

def sort_key(name):
    return (
        name.startswith('_'),
        -name.startswith('PyExc'),
        -name.endswith('Type'),
        name,
    )

new_entries = tomlkit.table()

def item(value):
    result = tomlkit.item(value)
    result.trivia.trail = ''
    return result.indent(4)

def mark_abi_only(name):
    manifest['data'][name]['abi_only'] = item(True)


for name in sorted(manifest['data'], key=sort_key) + ['PyMethod_Type']:
    if name.startswith('_'):
        print(f'ignoring private {name}')
        mark_abi_only(name)
        continue
    if (name.endswith(('Iter_Type', 'IterItem_Type', 'IterKey_Type', 'IterValue_Type'))
        or name.endswith(('Keys_Type', 'Items_Type', 'Values_Type'))
    ):
        print(f'ignoring iter type {name}')
        mark_abi_only(name)
        continue
    if name in {
        'PyExc_EnvironmentError',
        'PyExc_IOError',
        'PyExc_WindowsError',
    }:
        print(f'ignoring OSError alias {name}')
        continue
    if name in {
        'PyModuleDef_Type',
    }:
        print(f'ignoring deprecated type {name}')
        continue
    if name in {
        'PyOS_InputHook',
        'PyStructSequence_UnnamedField',
        'Py_FileSystemDefaultEncodeErrors',
        'Py_FileSystemDefaultEncoding',
        'Py_HasFileSystemDefaultEncoding',
        'Py_UTF8Mode',
        'Py_Version',
    }:
        print(f'ignoring non-PyObject {name}')
        mark_abi_only(name)
        continue
    const_name = f'Py_CONSTANT_{name.upper()}'
    if name.startswith('PyExc_'):
        const_name = f'Py_CONSTANT_{name.removeprefix('Py')}'
    if name.endswith('Type'):
        const_name = f'Py_CONSTANT_{name
                                         .removeprefix('Py_')
                                         .removeprefix('Py')
                                         or 'TYPE'}'
    entry = tomlkit.table()
    entry.raw_append('added', item('3.16'))
    entry.raw_append('value', item(10 + len(new_entries)))

    if name in manifest['data']:
        entry.raw_append('legacy_constant', item(name))
        mark_abi_only(name)

    entry.invalidate_display_name()
    new_entries.raw_append(const_name, entry)



for name, entry in new_entries.items():
    print(name, entry)

with open('Misc/stable_abi.toml.new', 'w') as f:
    tomlkit.dump(manifest, f)
    tomlkit.dump({'const': new_entries}, f)

with open('_defines.txt', 'w') as f:
    for name, entry in new_entries.items():
        print(f'#define {name} {entry['value']}', file=f)

with open('_arraycontents.txt', 'w') as f:
    for name, entry in new_entries.items():
        assert 'ifdef' not in manifest['data'].get(entry['legacy_constant'], {})
        if name.startswith('Py_CONSTANT_Exc'):
            print(f'    NULL,  // {name}', file=f)
        else:
            print(f'    (PyObject*)(&{entry['legacy_constant']}),', file=f)

with open('_setupcontents.txt', 'w') as f:
    for name, entry in new_entries.items():
        if name.startswith('Py_CONSTANT_Exc'):
            print(f'    constants[{name}] = {entry['legacy_constant']};', file=f)

with open('_redefinitions.txt', 'w') as f:
    for name, entry in new_entries.items():
        if name.startswith('Py_CONSTANT_Exc'):
            star = ''
        else:
            star = '*'
        print(f'#define {entry['legacy_constant']} ({star}Py_GetConstantBorrowed({name}))', file=f)

with open('_doc.txt', 'w') as f:
    for name, entry in new_entries.items():
        print(f'      - * .. c:macro:: {name}', file=f)
        print(f'        * 3.16', file=f)
        print(f'        * ``{entry['value']}``', file=f)
        data_name = entry['legacy_constant']
        val = f':c:data:`{data_name}`'
        if data_name.startswith('PyExc_'):
            val = f':py:type:`{data_name.removeprefix('PyExc_')}`'
        print('        *', val, file=f)
