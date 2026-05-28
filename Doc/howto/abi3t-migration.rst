.. highlight:: c

.. _abi3t-migration-howto:

******************************************************
Migrating to Stable ABI for Free Threading (``abi3t``)
******************************************************

Starting with the 3.15 release, CPython supports a variant of the Stable ABI
that supports :term:`free-threaded <free threading>` Python:
Stable ABI for Free-Threaded Builds, or ``abi3t`` for short.
This document describes how to adapt C API extensions to support free threading.

Why do this
===========

The typical reason to use Stable ABI is to reduce the number of artifacts that
you need to build and distribute for each version of your library.

Without the Stable ABI, you must build a separate shared library, and typically
a *wheel* distribution, for each feature version of CPython you wish
to support.
For example, each tag in the following table represents a separate
library/wheel:

+-----------------+-------------------+------------------+
| CPython version | Non-free-threaded | Free-threaded    |
+=================+===================+==================+
| 3.12            | ``cpython-312``   | ---              |
+-----------------+-------------------+------------------+
| 3.13            | ``cpython-313``   | ``cpython-313t`` |
+-----------------+-------------------+------------------+
| 3.14            | ``cpython-314``   | ``cpython-314t`` |
+-----------------+-------------------+------------------+
| 3.15            | ``cpython-315``   | ``cpython-315t`` |
+-----------------+-------------------+------------------+
| 3.16            | ``cpython-316``   | ``cpython-316t`` |
+-----------------+-------------------+------------------+
| Future versions | (etc.)            | (etc.)           |
+-----------------+-------------------+------------------+

That's a lot of builds, especially when multiplied by the number
of supported platforms.

With the Stable ABI (``abi3``, introduced in CPython 3.2), a single extension
(per platform) can cover all *non-free-threaded* builds of CPython:

+-----------------+-------------------+------------------+
| CPython version | Non-free-threaded | Free-threaded    |
+=================+===================+==================+
| 3.12            | ``abi3``          | ---              |
+-----------------+                   +------------------+
| 3.13            |                   | ``cpython-313t`` |
+-----------------+                   +------------------+
| 3.14            |                   | ``cpython-314t`` |
+-----------------+                   +------------------+
| 3.15            |                   | ``cpython-315t`` |
+-----------------+                   +------------------+
| 3.16            |                   | ``cpython-316t`` |
+-----------------+                   +------------------+
| Future versions |                   | (etc.)           |
+-----------------+-------------------+------------------+

The Stable ABI for free-threaded builds (``abi3t``), introduced in
CPython 3.15, does the same for free-threaded builds.
And it's compatible with non-free-threaded ones as well:

+-----------------+-------------------+------------------+
| CPython version | Non-free-threaded | Free-threaded    |
+=================+===================+==================+
| 3.12            | ``abi3`` *        | ---              |
+-----------------+                   +------------------+
| 3.13            |                   | ``cpython-313t`` |
+-----------------+                   +------------------+
| 3.14            |                   | ``cpython-314t`` |
+-----------------+-------------------+------------------+
| 3.15            | ``abi3t``                            |
+-----------------+                                      +
| 3.16            |                                      |
+-----------------+                                      +
| Future versions |                                      |
+-----------------+-------------------+------------------+

\* (As above, the ``abi3`` extension is compatible with all non-free-threaded
builds; even the 3.15+ ones that this table "attributes" to ``abi3t``.)

Why *not* do this
-----------------

There are two main downsides to Stable ABI.

First, you extension may become slower, since Stable ABI prioritizes
compatibility over performance.
The difference is usually not noticeable, and often can be mitigated by
using the same source to build both a Stable ABI build and a few
version-specific ones for "tier 1" CPython versions.

Second, not all of the C API is available.
Extensions need to be ported to build for Stable ABI, which may be difficult
or, in rare cases, impossible.

Specifically, ``abi3t`` requires on API added in CPython 3.15.
If you want to build your extension for older versions of CPython from the
same source, you have two main options:

- Use preprocessor conditionals.

  When following this guide, use ``#ifdef Py_TARGET_ABI3T`` blocks whenever
  you are told to do a change that breaks the build on CPython versions you
  care about. Keep the pre-existing code in ``#else`` blocks.

  For hand-written C extensions, this approach is reasonable down to
  CPython 3.12 and above.
  Keeping compatibility with 3.11 and below is likely only worth it for code
  generators (like Cython, for example).

- Do not port to ``abi3t``, and continue building separate extensions for
  each version of CPython, until you can drop support for the older versions.

  This is a valid approach. Not all extensions need to switch to ``abi3t``
  right now.


Prerequisites
=============

This guide assumes that you have an extension written in C (or C++), which
you want to port to ``abi3t``.

Non-free-threaded Stable ABI
----------------------------

Your extension should support the Stable ABI (``abi3t``).
If not, either port it first, or be prepared to fix issues that this guide
will not mention.

Free-threading support
----------------------

Your extension should also support free-threaded builds.
If it does not, follow :ref:`freethreading-extensions-howto` first.

.. seealso::

   `Porting Extension Modules to Support Free-Threading
   <https://py-free-threading.github.io/porting/>`_:
   A community-maintained porting guide for extension authors.

Isolating Extension Modules
---------------------------

Your module should use :ref:`multi-phase initialization <multi-phase-initialization>`,
and it should either be isolated or limit itself to be loaded at most once
per process.
If it is not your case, follow :ref:`isolating-extensions-howto` first.
(See the :ref:`opt-out section <isolating-extensions-optout>` for the quick
way to get compliant.)


Setting up the build
====================

If you use a build tool (such as setuptools, meson-python, scikit-build-core),
search its documentation for a way to select ``abi3t``.
At the time of writing, not all of them have one; but if your tool does,
use it.
You may want to verify that it set the right flag by temporarily adding the
following just after ``#include <Python.h>``::

   #if Py_TARGET_ABI3T+0 <= 0x30f0000
   #error "abt3t define is not set!"
   #endif

This should result in a *different* error than `abt3t define is not set`.

.. note::

   If your tool doesn't support ``abi3t`` yet, set the following macro before
   including ``Python.h``::

      #define Py_TARGET_ABI3T 0x30f0000

   or specify it as a compiler flag, for example::

      -DPy_TARGET_ABI3T=0x30f0000

   Once your extension builds with this setting, it will be compatible with
   CPython 3.15 and above.
   See :ref:`abi3-compiling` for constants to select different minimum version.

   Setting the macro manually is not recommended.
   Build tools should take care of tagging the resulting extension properly --
   and if you don't use a tool, you'll need to tag manually as well.

This guide will ask you to do a series of changes.
After each one, verify that your extension still builds in the original
(non-``abi3t``) configuration, and ideally run tests on all supported Python
versions.
This will ensure that nothing breaks as you are porting.


Module export hook
==================

Unless you've done this step already, your extension module defines a
:ref:`module initialization function <extension-pyinit>`
named :samp:`PyInit_{<module_name>}`.
You will need to port it to :ref:`module export hook <extension-export-hook>`,
:samp:`PyModExport_{<module name>}`, a feature added in CPython 3.15 in
:pep:`793`.

Your init function should look like this (with ``<modname>`` and ``<moddef>``
replaced by your values):

.. code-block::
   :class: bad

   PyMODINIT_FUNC
   PyInit_<modname>(void)
   {
       return PyModuleDef_Init(&<moddef>);
   }

If it does not, get it to this shape.
This guide unfortunately cannot give you specific advice, but the
:ref:`PyInit documentation <extension-pyinit>` may be helpful.
If there is some code before the ``return``, move it to
a :c:macro:`Py_mod_create` or :c:macro:`Py_mod_exec` slot function;
if you cannot, leave it in for now.

The function references a ``PyModuleDef`` object (``<moddef>`` in the code
above).
Its definition should be similar to the following, with different values
and perhaps some fields unnnamed or left out:

.. code-block::
   :class: bad

   static PyModuleDef <moddef> = {
       PyModuleDef_HEAD_INIT,
       .m_name = "my_module",
       .m_doc = "my docstring",
       .m_size = sizeof(my_state_struct),
       .m_methods = my_methods,
       .m_slots = my_slots,
       .m_traverse = my_traverse,
       .m_clear = my_clear,
       .m_free = my_free,
   };

Remove this definition and the ``PyInit`` function (or put them in
an ``#ifndef Py_TARGET_ABI3T`` block, to retain backwards compatibility),
and replace them with the following:

.. code-block::
   :class: good

   PyABIInfo_VAR(abi_info);

   static PySlot my_slot_array[] = {
      PySlot_STATIC_DATA(Py_mod_abi, &abi_info),
      PySlot_STATIC_DATA(Py_mod_name, "my_module"),
      PySlot_STATIC_DATA(Py_mod_doc, "my docstring"),
      PySlot_SIZE(Py_mod_state_size, sizeof(my_state_struct)),
      PySlot_STATIC_DATA(Py_mod_methods, my_methods),
      PySlot_STATIC_DATA(Py_mod_slots, my_slots),
      PySlot_FUNC(Py_mod_traverse, my_traverse),
      PySlot_FUNC(Py_mod_clear, my_clear),
      PySlot_FUNC(Py_mod_free, my_free),
      PySlot_END
   }

   PyMODEXPORT_FUNC
   PyModExport_<modname>(void)
   {
       return my_slot_array;
   }

Leave out any fields that were missing, and substitute your own values.

See :c:type:`PySlot` and :c:ref:`export hook <extension-export-hook>`
documentation for details on this API.

Associated ``PyModuleDef``
--------------------------

When using the new API, a :c:type:`!PyModuleDef` structure will not be
associated with the resulting module -- we're not using that structure any
more.

Check your code for any of the following functions:

- :c:func:`PyModule_GetDef`
- :c:func:`PyType_GetModuleByDef`

If you do not use these, skip the rest of this section.

These functions are typically used for two purposes:

1. To get the definition the module was created with.
   This is no longer possible using the new API, as modules don't keep
   a reference to their “source material”.
   You will need to figure out a different way to pass the
   relevant data around.

.. _abi3t-migration-module-token:

2. To check if a given module object is “yours”.
   This use case is now served by :ref:`module tokens <ext-module-token>` --
   opaque pointers that identify a module.
   To use a token, declare (or reuse) a unique static variable, for example:

   .. code-block::
      :class: good

      static char my_token;

   and add a pointer to it in a new entry to your ``PySlot`` array:

   .. code-block::
      :class: good
      :emphasize-lines: 3

      static PySlot my_slot_array[] = {
         ...
         PySlot_STATIC_DATA(Py_mod_token, &my_token),
         PySlot_END
      }

   Then, switch from :c:func:`PyModule_GetDef` calls such as:

   .. code-block::
      :class: bad

      PyModuleDef *def = PyModule_GetDef(module);

   to :c:func:`PyModule_GetToken` (which uses an output argument and may fail
   with an exception):

   .. code-block::
      :class: good

      void *token;
      if (PyModule_GetToken(module, &token) < 0) {
         /* handle error */
      }

   and from :c:func:`PyType_GetModuleByDef` calls such as:

   .. code-block::
      :class: bad

      PyObject *module = PyType_GetModuleByDef(type, my_def);
      /* use module */

   to :c:func:`PyType_GetModuleByToken` (which returns a strong reference):

   .. code-block::
      :class: good

      PyObject *module = PyType_GetModuleByToken(type, my_token);
      /* use module */
      Py_DECREF(module);

``PyObject`` opaqueness
=======================

The :c:type:`PyObject` and :c:type:`PyVarObject` structures are opaque
in ``abi3t``.

Accessing their members is prohibited.
If you do this, switch to getter/setter functions mentioned in
their documentation:

- :c:member:`PyObject.ob_type`
- :c:member:`PyObject.ob_refcnt`
- :c:member:`PyVarObject.ob_size`


Custom type definitions
-----------------------

Opaqueness means that the *size* of the :c:type:`PyObject` structure is
unknown to the compiler.
It can -- and *does* -- change between different CPython builds.

This means that the traditional way of defining custom types no longer works:

.. code-block::
   :class: bad

   typedef struct {
      PyObject_HEAD  // expands to `PyObject ob_base;` which has unknown size

      int my_data;
   } CustomObject;

   static PyTypeObject CustomType = {
      ...
      .tp_basicsize = sizeof(CustomObject),
      ...
   };

Most likely, all your class definitions, *and* all your access to class data,
will need to be rewritten.
This will probably be the biggest change you need to support ``abi3t``.

For each such type:

Instead of defining a ``struct`` for the entire instance, define one *only*
for the “additional” fields -- ones specific to your class, not its
superclasses:

.. code-block::
   :class: good

   typedef struct {
      int my_data;
   } CustomObjectData;

Change the name.
This is important: almost all code that uses the struct will need to change
-- notably, pointers to the new structure cannot be cast to/from ``PyObject*``.
The changed name will highlight the usages as compiler errors.
(If you use ``typeof``, C++ ``auto`` or similar ways to avoid
typing the type name, this won't work. Be extra careful, and consider running
tools to detect undefined behavior.)

Then, to create the class, use *negative* ``tp_basicsize`` to indicate
“extra” storage space rather than *total* instance size:

.. code-block::
   :class: good

   static PyTypeObject CustomType = {
      ...
      .tp_basicsize = -sizeof(CustomObjectData),
      ...
   };

If you use :c:macro:`Py_tp_members`, set the :c:macro:`Py_RELATIVE_OFFSET`
flag on each member and specify the :c:member:`~PyMemberDef.offset`
relative to your new struct.


Custom type data access
-----------------------

Then comes the hard part: in all code that needs to access this struct,
you will need an additional call to retrieve ``CustomObjectData *``
from ``PyObject *``:


.. code-block::
   :class: good

   PyObject *obj = ...;
   CustomObjectData *data = PyObject_GetTypeData(obj, cls);

Note that this call requires ``cls`` -- the *type object* for your class.

If your class is not subclassable (that is, it does not use the
:c:macro:`Py_TPFLAGS_BASETYPE` flag), this will be ``Py_TYPE(obj)``.
Otherwise, **DO NOT USE** ``Py_TYPE`` with :c:func:`!PyObject_GetTypeData`!
You will get memory reserved to *a subclass*.
For example, if a user makes a subclass like this:

.. code-block:: python

   class Sub(YourCustomClass):
      __slots__ = ('a', 'b')

the underlying memory may look like this:

.. code-block:: text

   ╭─ PyObject *obj;
   │            ╭─ the pointer you want
   │            │                  ╭─ PyObject_GetTypeData(obj, Py_TYPE(obj))
   ▼            ▼                  ▼
   ┌──────────┬─┬────────────────┬─┬─────────────┬─┬─────────────┐
   │ PyObject │*│ CustomTypeData │*│ PyObject *a │*│ PyObject *b │
   └──────────┴─┴────────────────┴─┴─────────────┴─┴─────────────┘

(Asterisks indicate possible padding.
Note that this memory layout is not guaranteed: future versions of Python may
add different padding or even switch the order of the structures.)

There are two main ways to get the right class:

- In instance methods, your implementation may use the :c:type:`PyCMethod`
  signature (and the :c:macro:`Py_METH_METHOD` :c:member:`PyMethodDef.ml_flags`
  bit), and get the class as the ``defining_class`` argument.
- Otherwise, give your class a unique static token using the
  :c:macro:`Py_tp_token` slot, and use:

  .. code-block::
     :class: good

     PyTypeObject cls;
     if (PyType_GetBaseByToken(Py_TYPE(obj), my_tp_token, &cls) < 0) {
         /* handle error */
     }
     CustomObjectData *data = PyObject_GetTypeData(obj, cls);

  Type tokens work similarly to module tokens covered :ref:`earlier in this
  guide <abi3t-migration-module-token>`.



Avoid build-time conditionals
=============================

Check your code for API that identifies the version of Python used to
*build* your extension.
This no longer corresponds to the Python your extension runs on, so code
code that uses this information often needs changing.
The most common are:

- :c:macro:`Py_GIL_DISABLED`: usually unneeded -- your code should
  work the same regardless of whether GIL is on or off.
- :c:macro:`PY_VERSION_HEX`:

  - use :c:data:`Py_Version` to get the run-time version;
  - use :c:macro:`Py_TARGET_ABI3T` to determine what C API is available
    (this macro is set to the minimum supported version).



Further code changes
====================

If you are left with compiler errors or warnings, fix them.
Alas, this guide is limited, and cannot cover all possible code
changes extensions may need.

If you find an issue, especially one that other extension authors might
run into, consider submitting an issue (or pull request) for this guide.

It is possible your issue cannot be fixed for the current version of ``abi3t``.
In that case, reporting it may help it get prioritized for the next version
of CPython.


Tagging and distribution
========================

If you are using a build tool with ``abi3t`` support, your extension is ready,
but you might want to check that it was built correctly.
If not, you have some renaming and tagging to do.

Extensions built with ``abi3t`` should have the following extension:

- On Windows: ``.pyd`` (like any other extension);
- Linux, macOS, and other systems that use the ``.so`` suffix: ``.abi3t.so``
  (**not** ``.cpython-315t.so`` or ``.abi3.so``).
  Note that both free-threaded and non-free-threaded builds will
  load ``.abi3t.so`` extensions;
- Other systems: consult your distributor, and perhaps update this guide.

If you distribute the extension as a *wheel*, use the following tags:

* Python tag: :samp:`cp3{YY}`, where *YY* is the minimum Python version
  the extension is built for.
  (For example, ``cp315`` if you set ``Py_TARGET_ABI3T`` to ``0x30f0000``.
  See :ref:`abi3-compiling` for more values.)
* ABI tag: ``abi3.abi3t``. This is a *compressed tag set* that indicates
  support for both non-free-threaded and free-threaded builds.

For example, the wheel filename may look like this:

.. code-block:: text

   myproject-1.0-cp315-abi3.abi3t-macosx_11_0_arm64.whl

.. seealso:: `Platform Compatibility Tags <https://packaging.python.org/en/latest/specifications/platform-compatibility-tags/>`__ in the PyPA package distribution metadata.


Testing
=======

Note that when you build an extension compatible with multiple versions of
CPython, you should always *test* it witch each version it supports (for
example, 3.15, 3.16, and so on).
Stable ABI only guarantees *ABI* compatibility; there may also be behavior
changes -- both intentional ones (covered by :pep:`387`) and bugs.

Be sure to run tests on both free-threaded and non-free-threaded builds
of CPython..
