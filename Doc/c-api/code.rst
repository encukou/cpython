.. highlight:: c

.. index:: object; code, code object

.. _codeobjects:

Code Objects
------------

.. sectionauthor:: Jeffrey Yasskin <jyasskin@gmail.com>

Code objects are a low-level detail of the CPython implementation.
Each one represents a chunk of executable code that hasn't yet been
bound into a function.

.. c:type:: PyCodeObject

   The C structure of the objects used to describe code objects.  The
   fields of this type are subject to change at any time.


.. c:var:: PyTypeObject PyCode_Type

   This is an instance of :c:type:`PyTypeObject` representing the Python
   :class:`code` type.


.. c:function:: int PyCode_Check(PyObject *co)

   Return true if *co* is a :class:`code` object.  This function always succeeds.

.. c:function:: int PyCode_GetNumFree(PyCodeObject *co)

   Return the number of free variables in *co*.

.. c:function:: PyCodeObject* PyUnstable_Code_New(int argcount, int kwonlyargcount, int nlocals, int stacksize, int flags, PyObject *code, PyObject *consts, PyObject *names, PyObject *varnames, PyObject *freevars, PyObject *cellvars, PyObject *filename, PyObject *name, int firstlineno, PyObject *linetable, PyObject *exceptiontable)

   Return a new code object.  If you need a dummy code object to create a frame,
   use :c:func:`PyCode_NewEmpty` instead.

   Since the definition of the bytecode changes often, calling
   :c:func:`PyCode_New` directly can bind you to a precise Python version.
   This function is  part of the semi-stable C API.
   See :c:macro:`Py_USING_SEMI_STABLE_API` for usage.

   The many arguments of this function are inter-dependent in complex
   ways, meaning that subtle changes to values are likely to result in incorrect
   execution or VM crashes. Use this function only with extreme care.

   .. versionchanged:: 3.11
      Added ``exceptiontable`` parameter.

   .. index:: single: PyCode_New

   .. versionchanged:: 3.12

      Renamed from ``PyCode_New`` as part of :ref:`unstable-c-api`.
      The old name is deprecated, but will remain available until the
      signature changes again.

.. c:function:: PyCodeObject* PyUnstable_Code_NewWithPosOnlyArgs(int argcount, int posonlyargcount, int kwonlyargcount, int nlocals, int stacksize, int flags, PyObject *code, PyObject *consts, PyObject *names, PyObject *varnames, PyObject *freevars, PyObject *cellvars, PyObject *filename, PyObject *name, int firstlineno, PyObject *linetable, PyObject *exceptiontable)

   Similar to :c:func:`PyCode_New`, but with an extra "posonlyargcount" for positional-only arguments.
   The same caveats that apply to ``PyCode_New`` also apply to this function.

   .. index:: single: PyCode_NewWithPosOnlyArgs

   .. versionadded:: 3.8 as ``PyCode_NewWithPosOnlyArgs``

   .. versionchanged:: 3.11
      Added ``exceptiontable`` parameter.

   .. versionchanged:: 3.12

      Renamed to ``PyUnstable_Code_NewWithPosOnlyArgs``.
      The old name is deprecated, but will remain available until the
      signature changes again.

.. c:function:: PyCodeObject* PyCode_NewEmpty(const char *filename, const char *funcname, int firstlineno)

   Return a new empty code object with the specified filename,
   function name, and first line number. The resulting code
   object will raise an ``Exception`` if executed.

.. c:function:: int PyCode_Addr2Line(PyCodeObject *co, int byte_offset)

    Return the line number of the instruction that occurs on or before ``byte_offset`` and ends after it.
    If you just need the line number of a frame, use :c:func:`PyFrame_GetLineNumber` instead.

    For efficiently iterating over the line numbers in a code object, use `the API described in PEP 626
    <https://peps.python.org/pep-0626/#out-of-process-debuggers-and-profilers>`_.

.. c:function:: int PyCode_Addr2Location(PyObject *co, int byte_offset, int *start_line, int *start_column, int *end_line, int *end_column)

   Sets the passed ``int`` pointers to the source code line and column numbers
   for the instruction at ``byte_offset``. Sets the value to ``0`` when
   information is not available for any particular element.

   Returns ``1`` if the function succeeds and 0 otherwise.

.. c:function:: PyObject* PyCode_GetCode(PyCodeObject *co)

   Equivalent to the Python code ``getattr(co, 'co_code')``.
   Returns a strong reference to a :c:type:`PyBytesObject` representing the
   bytecode in a code object. On error, ``NULL`` is returned and an exception
   is raised.

   This ``PyBytesObject`` may be created on-demand by the interpreter and does
   not necessarily represent the bytecode actually executed by CPython. The
   primary use case for this function is debuggers and profilers.

   .. versionadded:: 3.11

.. c:function:: PyObject* PyCode_GetVarnames(PyCodeObject *co)

   Equivalent to the Python code ``getattr(co, 'co_varnames')``.
   Returns a new reference to a :c:type:`PyTupleObject` containing the names of
   the local variables. On error, ``NULL`` is returned and an exception
   is raised.

   .. versionadded:: 3.11

.. c:function:: PyObject* PyCode_GetCellvars(PyCodeObject *co)

   Equivalent to the Python code ``getattr(co, 'co_cellvars')``.
   Returns a new reference to a :c:type:`PyTupleObject` containing the names of
   the local variables that are referenced by nested functions. On error, ``NULL``
   is returned and an exception is raised.

   .. versionadded:: 3.11

.. c:function:: PyObject* PyCode_GetFreevars(PyCodeObject *co)

   Equivalent to the Python code ``getattr(co, 'co_freevars')``.
   Returns a new reference to a :c:type:`PyTupleObject` containing the names of
   the free variables. On error, ``NULL`` is returned and an exception is raised.

   .. versionadded:: 3.11


Extra information
-----------------

To support low-level extensions to frame evaluation, such as external
just-in-time compilers, it is possible to attach arbitrary extra data to
code objects.

This functionality is a CPython implementation detail, and the API
may change without deprecation warnings.
These functions are part of the semi-stable C API.
See :c:macro:`Py_USING_SEMI_STABLE_API` for details.

See :pep:`523` for motivation and initial specification behind this API.


.. c:function:: Py_ssize_t PyUnstable_Eval_RequestCodeExtraIndex(freefunc free)

   Return a new an opaque index value used to adding data to code objects.

   You generally call this function once (per interpreter) and use the result
   with ``PyCode_GetExtra`` and ``PyCode_SetExtra`` to manipulate
   data on individual code objects.

   If *free* is not ``NULL``: when a code object is deallocated,
   *free* will be called on non-``NULL`` data stored under the new index.
   Use :c:func:`Py_DecRef` when storing :c:type:`PyObject`.

   Part of the semi-stable API, see :c:macro:`Py_USING_SEMI_STABLE_API`
   for usage.

   .. index:: single: _PyEval_RequestCodeExtraIndex

   .. versionadded:: 3.6 as ``_PyEval_RequestCodeExtraIndex``

   .. versionchanged:: 3.12

     Renamed to ``PyUnstable_Eval_RequestCodeExtraIndex``.
     The old private name is deprecated, but will be available until the API
     changes.

.. c:function:: int PyUnstable_Code_GetExtra(PyObject *code, Py_ssize_t index, void **extra)

   Set *extra* to the extra data stored under the given index.
   Return 0 on success. Set an exception and return -1 on failure.

   If no data was set under the index, set *extra* to ``NULL`` and return
   0 without setting an exception.

   Part of the semi-stable API, see :c:macro:`Py_USING_SEMI_STABLE_API`
   for usage.

   .. index:: single: _PyCode_GetExtra

   .. versionadded:: 3.6 as ``_PyCode_GetExtra``

   .. versionchanged:: 3.12

     Renamed to ``PyUnstable_Code_GetExtra``.
     The old private name is deprecated, but will be available until the API
     changes.

.. c:function:: int PyUnstable_Code_SetExtra(PyObject *code, Py_ssize_t index, void *extra)

   Set the extra data stored under the given index to *extra*.
   Return 0 on success. Set an exception and return -1 on failure.

   Part of the semi-stable API, see :c:macro:`Py_USING_SEMI_STABLE_API`
   for usage.

   .. index:: single: _PyCode_SetExtra

   .. versionadded:: 3.6 as ``_PyCode_SetExtra``

   .. versionchanged:: 3.12

     Renamed to ``PyUnstable_Code_SetExtra``.
     The old private name is deprecated, but will be available until the API
     changes.
