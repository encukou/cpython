#ifndef Py_INTERNAL_CALL_H
#define Py_INTERNAL_CALL_H
#ifdef __cplusplus
extern "C" {
#endif

#ifndef Py_BUILD_CORE
#  error "this header requires Py_BUILD_CORE define"
#endif

PyAPI_FUNC(PyObject *) _PyObject_Call_Prepend(
    PyThreadState *tstate,
    PyObject *callable,
    PyObject *obj,
    PyObject *args,
    PyObject *kwargs);

PyAPI_FUNC(PyObject *) _PyObject_FastCallDictTstate(
    PyThreadState *tstate,
    PyObject *callable,
    PyObject *const *args,
    size_t nargsf,
    PyObject *kwargs);

PyAPI_FUNC(PyObject *) _PyObject_Call(
    PyThreadState *tstate,
    PyObject *callable,
    PyObject *args,
    PyObject *kwargs);

static inline PyObject *
_PyObject_CallNoArgTstate(PyThreadState *tstate, PyObject *func) {
    return _PyObject_VectorcallTstate(tstate, func, NULL, 0, NULL);
}

PyObject *const *
_PyStack_UnpackDict(PyThreadState *tstate,
                    PyObject *const *args, Py_ssize_t nargs,
                    PyObject *kwargs, PyObject **p_kwnames);

void
_PyStack_UnpackDict_Free(PyObject *const *stack, Py_ssize_t nargs,
                         PyObject *kwnames);

#ifdef __cplusplus
}
#endif
#endif /* !Py_INTERNAL_CALL_H */
