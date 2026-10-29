#ifndef Py_CPYTHON_IMPORT_H
#  error "this header file must not be included directly"
#endif

// This structure is included in the public documentation
// (Doc/c-api/import.rst).
typedef struct PyImport_BuiltinInfo {
    const char *bi_name;
    uint16_t bi_type;
    uint16_t bi_flags;
    _Py_ANONYMOUS union {
        uint32_t bi_reserved;
    };
    _Py_ANONYMOUS union {
        PySlot *bi_slots;
        PySlot *(*bi_exportfunc)(void);
        PyObject* (*bi_initfunc)(void);
    };
} PyImport_BuiltinInfo;

// For PyImport_BuiltinInfo.bi_type:
#define Py_BUILTINMOD_END 0
#define Py_BUILTINMOD_SLOTS 1
#define Py_BUILTINMOD_MODEXPORT 2
#define Py_BUILTINMOD_INITFUNC 3

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

/* This structure is included verbatim in the public documentation
   (Doc/c-api/import.rst). */
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
