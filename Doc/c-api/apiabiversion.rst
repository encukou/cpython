.. highlight:: c

.. _apiabiversion:

***********************
API and ABI Versioning
***********************

CPython exposes its version number in the following macros.
Note that these correspond to the version code is **built** with,
not necessarily the version used at **run time**.

See :ref:`stable` for a discussion of API and ABI stability across versions.

.. c:macro:: PY_MAJOR_VERSION

   The ``3`` in ``3.4.1a2``.

.. c:macro:: PY_MINOR_VERSION

   The ``4`` in ``3.4.1a2``.

.. c:macro:: PY_MICRO_VERSION

   The ``1`` in ``3.4.1a2``.

.. c:macro:: PY_RELEASE_LEVEL

   The ``a`` in ``3.4.1a2``.
   This can be ``0xA`` for alpha, ``0xB`` for beta, ``0xC`` for release
   candidate or ``0xF`` for final.

.. c:macro:: PY_RELEASE_SERIAL

   The ``2`` in ``3.4.1a2``. Zero for final releases.

.. c:macro:: PY_VERSION_HEX

   The Python version number encoded in a single integer.
   See :c:func:`Py_PACK_VERSION` for the encoding details.

   Use this for numeric comparisons, e.g. ``#if PY_VERSION_HEX >= Py_PACK_VER(3, 14)``.

   This version is also available via the symbol :c:var:`Py_Version`.

.. c:var:: const unsigned long Py_Version

   The Python runtime version number encoded in a single constant integer, with
   the same format as the :c:macro:`PY_VERSION_HEX` macro.
   This contains the Python version used at run time.

   .. versionadded:: 3.11

.. c:function:: uint32_t Py_PACK_VERSION(major, minor, micro, release_level, release_serial)

   Return the given version, encoded as a single 32-bit integer with
   the following structure:

   +-------+-------------+------------------+------------------------------+-----------------------+
   | Bytes | Bits [#be]_ | Argument         | Constant                     | Value for ``3.4.1a2`` |
   +=======+=============+==================+==============================+=======================+
   |   1   | 1-8         | *major*          | :c:macro:`PY_MAJOR_VERSION`  | ``0x03``              |
   +-------+-------------+------------------+------------------------------+-----------------------+
   |   2   | 9-16        | *minor*          | :c:macro:`PY_MINOR_VERSION`  | ``0x04``              |
   +-------+-------------+------------------+------------------------------+-----------------------+
   |   3   | 17-24       | *micro*          | :c:macro:`PY_MICRO_VERSION`  | ``0x01``              |
   +-------+-------------+------------------+------------------------------+-----------------------+
   |   4   | 25-28       | *release_level*  | :c:macro:`PY_RELEASE_LEVEL`  | ``0xA``               |
   +       +-------------+------------------+------------------------------+-----------------------+
   |       | 29-32       | *release_serial* | :c:macro:`PY_RELEASE_SERIAL` | ``0x2``               |
   +-------+-------------+------------------+------------------------------+-----------------------+

   .. [#be]

      Bit positions are given in big-endian order

   For example:

   +-------------+---------------------------------------+---------------------+
   | Version     | ``Py_PACK_VERSION`` call              | Hexadecimal integer |
   +=============+=======================================+=====================+
   | ``3.4.1a2`` | ``Py_PACK_VERSION(3, 4, 1, 0xA, 2)``  | ``0x030401a2``      |
   +-------------+---------------------------------------+---------------------+
   | ``3.10.0``  | ``Py_PACK_VERSION(3, 10, 0, 0xF, 0)`` | ``0x030a00f0``      |
   +-------------+---------------------------------------+---------------------+

   :c:func:`!Py_PACK_VERSION` is primarily a macro, but is also available as
   an exported function. All arguments are of type ``unsigned char``.

   .. versionadded:: 3.14

.. c:function:: uint32_t Py_PACK_VER(major, minor)

   Equivalent to :c:expr:`Py_PACK_VERSION(major, minor, 0, 0, 0)`.
   This does not correspond to any released version of CPython,
   but it is useful for comparisons.

   Like :c:func:`!Py_PACK_VERSION`, :c:func:`!Py_PACK_VER` is also available as
   an exported function.

   .. versionadded:: 3.14
