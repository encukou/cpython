/*[clinic input]
preserve
[clinic start generated code]*/

PyDoc_STRVAR(_hmacopenssl_new__doc__,
"new($module, /, key, *, digestmod)\n"
"--\n"
"\n"
"Return a new hmac object.");

#define _HMACOPENSSL_NEW_METHODDEF    \
    {"new", (PyCFunction)_hmacopenssl_new, METH_FASTCALL, _hmacopenssl_new__doc__},

static PyObject *
_hmacopenssl_new_impl(PyObject *module, Py_buffer *key,
                      const char *digestmod);

static PyObject *
_hmacopenssl_new(PyObject *module, PyObject **args, Py_ssize_t nargs, PyObject *kwnames)
{
    PyObject *return_value = NULL;
    static const char * const _keywords[] = {"key", "digestmod", NULL};
    static _PyArg_Parser _parser = {"y*$s:new", _keywords, 0};
    Py_buffer key = {NULL, NULL};
    const char *digestmod;

    if (!_PyArg_ParseStack(args, nargs, kwnames, &_parser,
        &key, &digestmod)) {
        goto exit;
    }
    return_value = _hmacopenssl_new_impl(module, &key, digestmod);

exit:
    /* Cleanup for key */
    if (key.obj) {
       PyBuffer_Release(&key);
    }

    return return_value;
}

PyDoc_STRVAR(_hmacopenssl_HMAC_copy__doc__,
"copy($self, /)\n"
"--\n"
"\n"
"Return a copy (“clone”) of the HMAC object.");

#define _HMACOPENSSL_HMAC_COPY_METHODDEF    \
    {"copy", (PyCFunction)_hmacopenssl_HMAC_copy, METH_NOARGS, _hmacopenssl_HMAC_copy__doc__},

static PyObject *
_hmacopenssl_HMAC_copy_impl(HmacObject *self);

static PyObject *
_hmacopenssl_HMAC_copy(HmacObject *self, PyObject *Py_UNUSED(ignored))
{
    return _hmacopenssl_HMAC_copy_impl(self);
}

PyDoc_STRVAR(_hmacopenssl_HMAC_update__doc__,
"update($self, /, msg)\n"
"--\n"
"\n"
"Update the HMAC object with msg.");

#define _HMACOPENSSL_HMAC_UPDATE_METHODDEF    \
    {"update", (PyCFunction)_hmacopenssl_HMAC_update, METH_FASTCALL, _hmacopenssl_HMAC_update__doc__},

static PyObject *
_hmacopenssl_HMAC_update_impl(HmacObject *self, Py_buffer *msg);

static PyObject *
_hmacopenssl_HMAC_update(HmacObject *self, PyObject **args, Py_ssize_t nargs, PyObject *kwnames)
{
    PyObject *return_value = NULL;
    static const char * const _keywords[] = {"msg", NULL};
    static _PyArg_Parser _parser = {"y*:update", _keywords, 0};
    Py_buffer msg = {NULL, NULL};

    if (!_PyArg_ParseStack(args, nargs, kwnames, &_parser,
        &msg)) {
        goto exit;
    }
    return_value = _hmacopenssl_HMAC_update_impl(self, &msg);

exit:
    /* Cleanup for msg */
    if (msg.obj) {
       PyBuffer_Release(&msg);
    }

    return return_value;
}

PyDoc_STRVAR(_hmacopenssl_HMAC_digest__doc__,
"digest($self, /)\n"
"--\n"
"\n"
"Return the digest of the bytes passed to the update() method so far.");

#define _HMACOPENSSL_HMAC_DIGEST_METHODDEF    \
    {"digest", (PyCFunction)_hmacopenssl_HMAC_digest, METH_NOARGS, _hmacopenssl_HMAC_digest__doc__},

static PyObject *
_hmacopenssl_HMAC_digest_impl(HmacObject *self);

static PyObject *
_hmacopenssl_HMAC_digest(HmacObject *self, PyObject *Py_UNUSED(ignored))
{
    return _hmacopenssl_HMAC_digest_impl(self);
}

PyDoc_STRVAR(_hmacopenssl_HMAC_hexdigest__doc__,
"hexdigest($self, /)\n"
"--\n"
"\n"
"Return hexadecimal digest of the bytes passed to the update() method so far.\n"
"\n"
"This may be used to exchange the value safely in email or other non-binary\n"
"environments.");

#define _HMACOPENSSL_HMAC_HEXDIGEST_METHODDEF    \
    {"hexdigest", (PyCFunction)_hmacopenssl_HMAC_hexdigest, METH_NOARGS, _hmacopenssl_HMAC_hexdigest__doc__},

static PyObject *
_hmacopenssl_HMAC_hexdigest_impl(HmacObject *self);

static PyObject *
_hmacopenssl_HMAC_hexdigest(HmacObject *self, PyObject *Py_UNUSED(ignored))
{
    return _hmacopenssl_HMAC_hexdigest_impl(self);
}
/*[clinic end generated code: output=10b6e8cac6d7a2c9 input=a9049054013a1b77]*/
