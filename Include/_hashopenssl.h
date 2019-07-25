#ifndef Py_HASHOPENSSL_H
#define Py_HASHOPENSSL_H

#include "Python.h"
#include <openssl/crypto.h>
#include <openssl/err.h>

/* LCOV_EXCL_START */
static PyObject *
_setException(PyObject *exc)
{
    unsigned long errcode;
    const char *lib, *func, *reason;

    errcode = ERR_peek_last_error();
    if (!errcode) {
        PyErr_SetString(exc, "unknown reasons");
        return NULL;
    }
    ERR_clear_error();

    lib = ERR_lib_error_string(errcode);
    func = ERR_func_error_string(errcode);
    reason = ERR_reason_error_string(errcode);

    if (lib && func) {
        PyErr_Format(exc, "[%s: %s] %s", lib, func, reason);
    }
    else if (lib) {
        PyErr_Format(exc, "[%s] %s", lib, reason);
    }
    else {
        PyErr_SetString(exc, reason);
    }
    return NULL;
}
/* LCOV_EXCL_STOP */


__attribute__((__unused__))
static int
_Py_hashlib_fips_error(char *name) {
    int result = FIPS_mode();
    if (result == 0) {
        // "If the library was built without support of the FIPS Object Module,
        // then the function will return 0 with an error code of
        // CRYPTO_R_FIPS_MODE_NOT_SUPPORTED (0x0f06d065)."
        // But 0 is also a valid result value.

        unsigned long errcode = ERR_peek_last_error();
        if (errcode) {
            _setException(PyExc_ValueError);
            return 1;
        }
        return 0;
    }
    PyErr_Format(PyExc_ValueError, "%s is not available in FIPS mode",
                 name);
    return 1;
}

#define FAIL_RETURN_IN_FIPS_MODE(name) do { \
    if (_Py_hashlib_fips_error(name)) return NULL; \
} while (0)

#endif  // !Py_HASHOPENSSL_H
