// To use preserve_none in JIT builds, we need to declare a separate function
// pointer with clang::preserve_none, since this attribute may not be
// supported by the compiler used to build the rest of the interpreter.
typedef jit_func [[clang::preserve_none]] jit_func_preserve_none;

#define PATCH_VALUE(TYPE, NAME, ALIAS) \
    PyAPI_DATA(void) ALIAS;            \
    TYPE NAME = (TYPE)(uintptr_t)&ALIAS;

#define DECLARE_TARGET(NAME)                     \
    _Py_CODEUNIT * [[clang::preserve_none, clang::visibility("hidden")]] \
    NAME(_PyExecutorObject *executor, _PyInterpreterFrame *frame, _PyStackRef *stack_pointer, PyThreadState *tstate, \
    _PyStackRef _tos_cache0, _PyStackRef _tos_cache1, _PyStackRef _tos_cache2);
