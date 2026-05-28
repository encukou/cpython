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
For example, each "tag" in the following table represents a separate
built artifacts:

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

The number of artifacts is multiplied by the number of supported platforms.
For example, the there are `88 wheels <https://pypi.org/project/MarkupSafe/3.0.3/#files>`_
for a single version of the ``markupsafe`` extension,
all built from the same source.

With the Stable ABI (``abi3``, introduced in CPython 3.2), a single extension
(per platform) for *all* non-free-threaded builds of CPython:

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

The Stable ABI for free-threaded builds (``abi3t``), introduced in CPython 3.15
does the same for free-threaded builds:

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
builds; in this table, 3.15+ are "covered" by ``abi3t``.)

Why *not* do this
-----------------

There are two main downsides to Stable ABI.

First, you extension may become slower, since Stable ABI prioritizes
compatibility over performance.
The difference is usually not noticeable, and often can be mitigated by
building version-specific extensions *in addition* to the Stable ABI one,
from the same source.

Second, not all of the C API is available.
Extensions need to be ported to build for Stable ABI, which may be difficult
or, in rare cases, impossible.

Specifically, ``abi3t`` relies on API added in CPython 3.15.
If your extension supports older versions of CPython, you have two main
options:

- Use preprocessor conditionals.

  When following this guide, use ``#ifdef Py_TARGET_ABI3T`` blocks whenever
  you are told to do a change that breaks the build on CPython versions you
  care about, and keep the pre-existing code in ``#else`` blocks.

  For hand-written C extensions, this approach is reasonable if you support
  CPython 3.12 and above. If you need to support 3.11 and below,
  it will likely be a frustrating experience.

  For code generators like Cython, 3.13

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
replaced by your values)::

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
and perhaps some fields unnnamed or left out::

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
and replace them with the following::

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

When using the new API, a ``PyModuleDef`` structure will not be associated
with the resulting module -- we're not using that structure any more.
Check your code for any of the following functions:

- :c:func:`PyModule_GetDef`
- :c:func:`PyType_GetModuleByDef`
- :c:func:`PyType_GetModuleByToken`

If you use any of these, add an additional entry to your ``PySlot`` array::

   static PySlot my_slot_array[] = {
      ...
      PySlot_STATIC_DATA(Py_mod_token, &mod_token),
      PySlot_END
   }



Tagging and distribution
========================


Note that when you build an extension compatible with multiple versions of
CPython, you should always *test* it witch each version it supports.
Stable ABI only guarantees *ABI* compatibility.


Prerequisites
=============

This guide assumes that you have an extension written in C (or C++), which
you want to port to ``abi3t``.


Why use `abi3t`, and what are the alternatives?
===============================================

An extension compiled for ``abi3t`` will be loadable on future versions of
CPython 3, without recompilation.
Such an extension will also be compatible with ``abi3`` (Stable ABI for
non-free-threaded builds).

For ``abi3``




For 3.14 and 3.13, continue compiling with the version-specific ABI. This document describes how
to adapt C API extensions to support free threading.

Identifying the Free-Threaded Limited API Build in C
====================================================

Define :c:macro:`!Py_TARGET_ABI3T` to the lowest Python version your extension supports,
either in the form of ``Py_PACK_VERSION(3.15)`` or its direct hex value (such as ``0x30f0000`` for 3.15).
You can use it to enable code that only runs under the free-threaded build::

    #ifdef Py_TARGET_ABI3T
    /* code that only runs in the free-threaded stable ABI build */
    #endif

``PyObject`` and ``PyVarObject`` opaqueness
===========================================

Accessing any member of ``PyObject`` directly is now prohibited, unlike the GIL
stable ABI, where accessing such members are merely discouraged.
For instance, prefer ``Py_TYPE()`` and ``Py_SET_TYPE()`` over ``ob_type``,
``Py_REFCNT``, ``Py_IncRef()`` and ``Py_DecRef()`` over ``ob_refcnt``, etc.
Also, embedding :c:macro:`PyObject_HEAD` within a struct is impossible.

Similarly, members of ``PyVarObject`` are not visible. If you need any object of such type
to be passed as a ``PyObject`` parameter to any API function, cast it directly as ``PyObject``.

Module Initialization
=====================

Extension modules need to explicitly indicate that they support running with
the GIL disabled; otherwise importing the extension will raise a warning and
enable the GIL at runtime.

Multi-phase and single-phase initialization is supported to indicate that an extension module
targeting the stable ABI supports running with the GIL disabled, though the former is preferred.

Multi-Phase Initialization
---------------------------

Extensions that use :ref:`multi-phase initialization <multi-phase-initialization>`
(functions like :c:func:`PyModuleDef_Init`,
:c:func:`PyModExport_* <PyModExport_modulename>` export hook,
:c:func:`PyModule_FromSlotsAndSpec`) should add a
:c:data:`Py_mod_gil` slot in the module definition.
If your extension supports older versions of CPython,
you should guard the slot with a :c:data:`Py_GIL_DISABLED` check.
Additionally, prefer :c:type:`PySlot` over :c:type:`PyModuleDef_Slot`.

::

    static PySlot module_slots[] = {
        ...
    #ifdef Py_GIL_DISABLED
        PySlot_STATIC_DATA(Py_mod_gil, Py_MOD_GIL_NOT_USED),
    #endif
        PySlot_END
    };

Furthermore, using ``PyABIInfo_VAR`` and ``Py_mod_abi`` is recommended so that an
extension module loaded for an incompatible interpreter will trigger an exception, rather than
fail with a crash.

.. code-block:: c

   #ifdef PY_VERSION_HEX >= 0x030F0000
   PyABIInfo_VAR(abi_info);
   #endif Py_GIL_DISABLED

   static PySlot mymodule_slots[] = {
      ...
   #ifdef PY_VERSION_HEX >= 0x030F0000
      PySlot_STATIC_DATA(Py_mod_abi, &abi_info),
   #endif
      PySlot_END
   };

Single-Phase Initialization
---------------------------

Although members of ``PyModuleDef`` is still available for no-GIL Stable ABI and can be used
for :ref:`single-phase initialization <single-phase-initialization>`
(that is, :c:func:`PyModule_Create`), they are not exposed when targeting the regular Stable ABI.
Prefer multi-phased initializtion when possible.


Critical Sections
=================

.. _critical-sections:

Equivalent functions:

+-------------------------------------------+---------------------------------------+
| Macro functions                           | C API functions                       |
+===========================================+=======================================+
|:c:macro:`Py_BEGIN_CRITICAL_SECTION`       |``PyCriticalSection_Begin``            |
|:c:macro:`Py_END_CRITICAL_SECTION`         |``PyCriticalSection_End``              |
+-------------------------------------------+---------------------------------------+
|:c:macro:`Py_BEGIN_CRITICAL_SECTION2`      |``PyCriticalSection2_Begin``           |
|:c:macro:`Py_END_CRITICAL_SECTION2`        |``PyCriticalSection2_End``             |
+-------------------------------------------+---------------------------------------+
|:c:macro:`Py_BEGIN_CRITICAL_SECTION_MUTEX` |``PyCriticalSection_BeginMutex``       |
|:c:macro:`Py_END_CRITICAL_SECTION`         |``PyCriticalSection_End``              |
+-------------------------------------------+---------------------------------------+
|:c:macro:`Py_BEGIN_CRITICAL_SECTION2_MUTEX`|``PyCriticalSection2_BeginMutex``      |
|:c:macro:`Py_END_CRITICAL_SECTION2`        |``PyCriticalSection2_End``             |
+-------------------------------------------+---------------------------------------+

Platform-specific considerations
--------------------------------

On some platforms, Python will look for and load shared library files named
with the ``abi3`` or ``abi3t`` tag (for example, ``mymodule.abi3t.so``).
:term:`Free-threaded <free-threaded build>` interpreters prefer ``abi3t``,
but can fall back to ``abi3``.
Thus, extensions compatible with both ABIs should use the ``abi3t`` tag.

Python does not necessarily check that extensions it loads
have compatible ABI.
Extension authors are encouraged to add a check using the :c:macro:`Py_mod_abi`
slot or the :c:func:`PyABIInfo_Check` function.

Limited C API Build Tools
-------------------------

If you use
`setuptools <https://setuptools.pypa.io/en/latest/setuptools.html>`_ to build
your extension, a future version of ``setuptools`` will allow ``py_limited_api=True``
to be set to allow targeting limited API when building with the free-threaded build.
``uv`` supports targeting PEP 803 as of 0.11.3: ``https://github.com/astral-sh/uv/releases/tag/0.11.3``.

`Other build tools will support this ABI as well <https://packaging.python.org/en/latest/guides/tool-recommendations/#build-backends-for-extension-modules>`_.

.. seealso::

   `Porting Extension Modules to Support Free-Threading
   <https://py-free-threading.github.io/porting/>`_:
   A community-maintained porting guide for extension authors.
