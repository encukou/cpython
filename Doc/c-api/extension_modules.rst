.. highlight:: c

.. _extension-modules:

Defining Extension Modules
--------------------------

A C extension for CPython is a shared library (e.g. a ``.so`` file on Linux,
``.pyd`` on Windows), which exports an
:ref:`initialization function <extension-export-hook>`.

To be importable by default (that is, by
:py:class:`importlib.machinery.ExtensionFileLoader`),
the shared library must be available on :py:attr:`sys.path`,
and must be named after the module name plus an extension listed in
:py:attr:`importlib.machinery.EXTENSION_SUFFIXES`.

.. note::

   Building, packaging and distributing extension modules is best done with
   third-party tools, and is out of scope of this document.
   One suitable tool is ``setuptools``, whose documentation can be found at
   https://setuptools.readthedocs.io/en/latest/setuptools.html.

The initialization function may return one of two values:

Normally, the initialization function returns a module definition initialized
using :c:func:`PyModuleDef_Init`.
This allows splitting the creation process into several phases:

- Before any substantial code is executed, Python can determine which
  capabilities the module supports, and adjust the environment or
  refuse loading it into an incompatible context.
- By default, Python itself creates the module object -- that is, it does
  the equivalent of :py:meth:`~object.__new__` for classes.
  It also sets initial attributes like :attr:`~module.__package__` and
  :attr:`~module.__loader__`.
- Afterwards, the module object is initialized using user code -- the
  equivalent of :py:meth:`~object.__init__` on classes.

This is called *multi-phase initialization* to distinguish it from the legacy
(but still supported) *single-phase initialization* scheme,
where the initialization function returns a fully constructed module.
See the :ref:`relevant section below <single-phase-initialization>` for details.

.. versionchanged:: 3.5

   Added support for multi-phase initialization (:pep:`489`).


Multiple module instances
.........................

By default, extension modules are not singletons.
For example, if the :py:attr:`sys.modules` entry is removed and the module
is re-imported, a new module object is created, and typically populated with
fresh method and type objects.
The old module is subject to normal garbage collection.
This mirrors the behavior of pure-Python modules.

Additional module instances may be created in
:ref:`sub-interpreters <sub-interpreter-support>`
or after after Python runtime reinitialization
(:c:func:`Py_Finalize` and :c:func:`Py_Initialize`).
In these cases, sharing Python objects between module instances would likely
cause crashes or undefined behavior.

To avoid such issues, each instance of an extension module should
be *isolated*: changes to one instance should not implicitly affect the others,
and all state owned by the module, including references to Python objects,
should be specific to a particular module instance.
See :ref:`isolating-extensions-howto` for more details and a practical guide.

A simpler way to avoid these issues is
:ref:`raising an error on repeated initialization <isolating-extensions-optout>`.

All modules are expected to support
:ref:`sub-interpreters <sub-interpreter-support>`, or otherwise explicitly
signal a lack of support.
This is usually achieved by isolation or blocking repeated initialization,
as above.
A module may also be limited to the main interpreter using
the :c:data:`Py_mod_multiple_interpreters` slot.


.. _extension-export-hook:

Initialization function
.......................

The initialization function, also known as *export hook*, defined by the
extension, has the signature:

.. c:function:: PyObject* PyInit_modulename(void)

To make sure Python can load the initialization function, it is recommended
to define it using :c:macro:`PyMODINIT_FUNC`.

Its name should be :samp:`PyInit_{<name>}`, with ``<name>`` replaced by the
name of the module.

For modules with ASCII-only names, the function must instead be named
:samp:`PyInit_{<name>}`, with ``<name>`` replaced by the name of the module.
When using :ref:`multi-phase-initialization`, non-ASCII module names
are allowed. In this case, the initialization function name is
:samp:`PyInitU_{<name>}`, with ``<name>`` encoded using Python's
*punycode* encoding with hyphens replaced by underscores. In Python:

.. code-block:: python

    def initfunc_name(name):
        try:
            suffix = b'_' + name.encode('ascii')
        except UnicodeEncodeError:
            suffix = b'U_' + name.encode('punycode').replace(b'-', b'_')
        return b'PyInit' + suffix

It is possible to export multiple modules from a single shared library by
defining multiple initialization functions. However, importing them requires
using symbolic links or a custom importer, because by default only the
function corresponding to the filename is found.
See the *"Multiple modules in one library"* section in :pep:`489` for details.


.. _multi-phase-initialization:

Multi-phase initialization
..........................

Normally, the :ref:`initialization function <extension-export-hook>`
(``PyInit_modulename``) returns a :c:type:`PyModuleDef` instance with
non-empty :c:member:`~PyModuleDef.m_slots`.
Before it is returned, the ``PyModuleDef`` instance must be initialized
using the following function:


.. c:function:: PyObject* PyModuleDef_Init(PyModuleDef *def)

   Ensure a module definition is a properly initialized Python object that
   correctly reports its type and a reference count.

   Return *def* cast to ``PyObject*``, or ``NULL`` if an error occurred.

   Calling this function is required for :ref:`multi-phase-initialization`.
   It should not be used other contexts.

   Note that Python assumes that ``PyModuleDef`` structures are statically
   allocated.
   The reference returned by this function must not be released.

   .. versionadded:: 3.5


.. _single-phase-initialization:

Legacy single-phase initialization
..................................

.. attention::
   Single-phase initialization is a legacy mechanism to initialize extension
   modules, with known drawbacks and design flaws. Extension module authors
   are encouraged to use multi-phase initialization instead.

In single-phase initialization, the
:ref:`initialization function <extension-export-hook>` (``PyInit_modulename``)
should create, populate and return a module object, typically using :c:func:`PyModule_Create` and functions like :c:func:`PyModule_AddObjectRef`.

Single-phase initialization differs from the default in the following ways:

* Single-phase modules are, or rather *contain*, “singletons”.

  When the module is initialized, Python saves the contents of the module's
  ``__dict__`` (that is, typically, the module's functions and types).

  For subsequent initializations in the same interpreter, Python does not call
  the initializatoin function again.
  Instead, it creates a new module object with a new ``__dict__``, and copies
  the saved contents to it.
  For example, given a single-phase module ``_testsinglephase``
  [#testsinglephase]_ that defines a function ``sum`` and an exception class
  ``error``:

  .. code-block:: python

     >>> import sys
     >>> import _testsinglephase as one
     >>> del sys.modules['_testsinglephase']
     >>> import _testsinglephase as two
     >>> one is two
     False
     >>> one.__dict__ is two.__dict__
     False
     >>> one.sum is two.sum
     True
     >>> one.error is two.error
     True

   The exact behavior should be considered a CPython implementation detail.

* Non-ASCII module names (``PyInitU_modulename``) are not supported.

* To work around the fact that ``PyInit_modulename`` does not take a *spec*
  argument, some state of the import machinery is saved and applied to the
  first suitable module created during the ``PyInit_modulename`` call.
  Specifically, when a sub-module is imported, this mechanism prepends the
  parent package name to the name of the module.

  A single-phase ``PyInit_modulename`` function should create “its” module
  object as soon as possible, before any other module objects can be created.

* Single-phase modules do not support slots, that is,
  :c:member:`~PyModuleDef.m_slots` must be ``NULL``.

* Single-phase modules support module lookup functions like
  :c:func:`PyState_FindModule`.

.. [#testsinglephase] ``_testsinglephase`` is an internal module used \
   in CPython's self-test suite; your distribution may or may not \
   include it.
