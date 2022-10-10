#define Py_LIMITED_API 0x030c0000 // 3.12
#include "parts.h"

#ifdef LIMITED_API_AVAILABLE

static PyMethodDef TestMethods[] = {
    /* Add module methods here.
     * (Empty list left here as template/example, since using
     * PyModule_AddFunctions isn't very common.)
     */
    {NULL},
};

int
_PyTestCapi_Init_HeaptypeRelative(PyObject *m) {
    if (PyModule_AddFunctions(m, TestMethods) < 0) {
        return -1;
    }

    return 0;
}

#endif // LIMITED_API_AVAILABLE
