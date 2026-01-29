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

struct PyModuleTab_Entry {
    const char *mt_name;
    uint16_t mt_type;
    uint16_t mt_flags;
    _Py_ANONYMOUS union {
        uint32_t _mt_reserved;
    };
    _Py_ANONYMOUS union {
        void *mt_ptr;
        const PyModuleDef_Slot *mt_slots;
        PyObject* (*mt_initfunc)(void);
    };
};

#define PyModuleTab_TYPE_END 0 // (end marker)
#define PyModuleTab_TYPE_SLOTS 1 // use mt_slots
#define PyModuleTab_TYPE_INITFUNC 2 // use mt_initfunc
#define _PyModuleTab_TYPE_SPECIAL 3 // (sys & builtins)
