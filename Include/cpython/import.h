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

typedef struct PyInittab2_Entry {
    const char *m_name;
    uint16_t m_type;
    uint16_t m_flags;
    uint16_t m_reserved;
    uint16_t m_typeflags;
    union {
        PyModuleDef_Slot *m_slots;
        PyObject* (*m_initfunc)(void);
        struct {
            const unsigned char *frz_code;
            Py_ssize_t frz_size;
        } m_frozen;
        struct _frozen *_m_internal_frozen;
    };
} PyInittab2_Entry;

#ifdef PyInittab2_USE_CUSTOM_IMPLEMENTATION
extern int PyInittab2_FindEntry(
    const char *name, struct PyInittab2_Entry *result);
extern int PyInittab2_NextEntry(
    const struct PyInittab2_Entry **entry);
extern int PyInittab2_FinishIteration(
    const struct PyInittab2_Entry *entry);
#else
PyAPI_FUNC(int) PyInittab2_FindEntry(
    const char *name, struct PyInittab2_Entry *result);
PyAPI_FUNC(int) PyInittab2_NextEntry(
    const struct PyInittab2_Entry **entry);
PyAPI_FUNC(int) PyInittab2_FinishIteration(
    const struct PyInittab2_Entry *entry);
#endif

PyAPI_FUNC(int) PyUnstable_Inittab2_Default_FindEntry(
    const char *name, struct PyInittab2_Entry *result);
PyAPI_FUNC(int) PyUnstable_Inittab2_Default_NextEntry(
    const struct PyInittab2_Entry **entry);
PyAPI_FUNC(int) PyUnstable_Inittab2_Default_FinishIteration(
    const struct PyInittab2_Entry *entry);
