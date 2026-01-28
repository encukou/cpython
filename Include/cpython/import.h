#ifndef Py_CPYTHON_IMPORT_H
#  error "this header file must not be included directly"
#endif

struct _inittab {
    const char *name;           /* ASCII encoded string */
    PyObject* (*initfunc)(void);
};
// This is not used after Py_Initialize() is called.
PyAPI_DATA(struct _inittab *) PyImport_Inittab;
PyAPI_FUNC(int) PyImport_ExtendInittab(struct _inittab *newtab);

// Custom importers may use this API to initialize statically linked
// extension modules directly from a spec and init function,
// without needing to go through inittab
PyAPI_FUNC(PyObject *) PyImport_CreateModuleFromInitfunc(
    PyObject *spec,
    PyObject *(*initfunc)(void));

struct _frozen {
    const char *name;                 /* ASCII encoded string */
    const unsigned char *code;
    int size;
    int is_package;
};

/* Embedding apps may change this pointer to point to their favorite
   collection of frozen modules: */

PyAPI_DATA(const struct _frozen *) PyImport_FrozenModules;

PyAPI_FUNC(PyObject*) PyImport_ImportModuleAttr(
    PyObject *mod_name,
    PyObject *attr_name);
PyAPI_FUNC(PyObject*) PyImport_ImportModuleAttrString(
    const char *mod_name,
    const char *attr_name);

typedef struct PyImportBuiltin_Entry_Meta {
    uint16_t m_datatype;
    uint16_t m_flags;
    _Py_ANONYMOUS union {
        uint32_t m_reserved;
    };
} PyImportBuiltin_Entry_Meta;

typedef struct PyImportBuiltin_Entry {
    PyImportBuiltin_Entry_Meta e_meta;
    _Py_ANONYMOUS union {
        void *e_ptr;
        const PyModuleDef_Slot *e_slots;
        PyObject* (*e_initfunc)(void);
        const struct _frozen *e_frozen;
    };
} PyImportBuiltin_Entry;

#define PyImportBuiltin_TYPE_INITFUNC 1 // m_initfunc
#define PyImportBuiltin_TYPE_SLOTS 2 // m_slots
#define PyImportBuiltin_TYPE_FROZEN 3 // m_frozen
#define PyImportBuiltin_TYPE_SPECIAL 4 // no data (sys & builtins)

#define PyImportBuiltin_KIND_FROZEN 1
#define PyImportBuiltin_KIND_BUILTIN 2
#define PyImportBuiltin_KIND_ALL 0xff

#ifdef PyImportBuiltin_USE_CUSTOM_IMPLEMENTATION
extern int PyImportBuiltin_FindEntry(
    const char *name,
    PyImportBuiltin_Entry *result,
    Py_ssize_t result_struct_size);
extern PyObject *PyImportBuiltin_GetNames(void);
#else
PyAPI_FUNC(int) PyImportBuiltin_FindEntry(
    const char *name,
    int kind,
    PyImportBuiltin_Entry *result,
    Py_ssize_t result_struct_size);
PyAPI_FUNC(PyObject *) PyImportBuiltin_GetNames(int kinds);
#endif

PyAPI_FUNC(int) PyUnstable_ImportBuiltin_FindEntry_Default(
    const char *name,
    int kind,
    PyImportBuiltin_Entry *result,
    Py_ssize_t result_struct_size);
PyAPI_FUNC(PyObject *) PyUnstable_ImportBuiltin_GetNames_Default(int kinds);
