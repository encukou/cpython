/* Test version macros in the limited API */

#define Py_LIMITED_API 0x030E0000  // Added in 3.14

#include "parts.h"
#include "clinic/version.c.h"
#include <stdio.h>

/*[clinic input]
module _testlimitedcapi
[clinic start generated code]*/
/*[clinic end generated code: output=da39a3ee5e6b4b0d input=2700057f9c1135ba]*/

/*[clinic input]
_testlimitedcapi.pack_version

    major: int
    minor: int
    micro: int
    level: int
    serial: int
    /
[clinic start generated code]*/

static PyObject *
_testlimitedcapi_pack_version_impl(PyObject *module, int major, int minor,
                                   int micro, int level, int serial)
/*[clinic end generated code: output=7cff84fd2f51743f input=cada96a4af05363d]*/
{
    uint32_t macro_result = Py_PACK_VERSION(major, minor, micro, level, serial);
#undef Py_PACK_VERSION
    uint32_t func_result = Py_PACK_VERSION(major, minor, micro, level, serial);

    assert(macro_result == func_result);
    return PyLong_FromUnsignedLong((unsigned long)func_result);
}

/*[clinic input]
_testlimitedcapi.pack_ver

    major: int
    minor: int
    /
[clinic start generated code]*/

static PyObject *
_testlimitedcapi_pack_ver_impl(PyObject *module, int major, int minor)
/*[clinic end generated code: output=d560a3682a08a5e1 input=8a5b76f707a17316]*/
{
    uint32_t macro_result = Py_PACK_VER(major, minor);
#undef Py_PACK_VER
    uint32_t func_result = Py_PACK_VER(major, minor);

    assert(macro_result == func_result);
    return PyLong_FromUnsignedLong((unsigned long)func_result);
}

static PyMethodDef TestMethods[] = {
    _TESTLIMITEDCAPI_PACK_VERSION_METHODDEF
    _TESTLIMITEDCAPI_PACK_VER_METHODDEF
    {NULL},
};

int
_PyTestLimitedCAPI_Init_Version(PyObject *m)
{
    if (PyModule_AddFunctions(m, TestMethods) < 0) {
        return -1;
    }
    return 0;
}
