.. _building:
.. _setuptools-index:
.. _install-index:

*****************************
Building C and C++ Extensions
*****************************

Building, packaging and distributing extension modules is best done with
third-party tools.
One suitable tool is ``setuptools``, whose documentation can be found at
https://setuptools.readthedocs.io/en/latest/setuptools.html.

The :mod:`distutils` module, which was removed from the standard library in
Python 3.12, is maintained as part of ``setuptools``.


A Manual Approach for Unix-like systems
=======================================

If you do not wish to use ``setuptools``, or are building a similar tool,
this section can offer some limited advice.
Note that there are many ways in which Python can be installed, and many
variations in platforms.
These notes may not fully apply to your particular case.

Python comes with a `pkg-config <https://en.wikipedia.org/wiki/Pkg-config>`__-compatible
script called, for example, ``python-config`` or ``python3.15-config``,
which may be installed by your distribution of CPython, or even integrated into
your system *pkg-config* database.
Be sure to use a version of this script that corresponds to the Python build
you are targetting.

The ``python-config`` script takes several options; the most relevant ones
being ``--cflags`` (C compiler flags, including header file locations)
and ``--ldflags`` (linker flags).
When compiling and linking a single source file, you may specify both.

Thus, the following command would be a strating point for manual compliation:

.. code-block:: console

   $ cc $(python-config --cflags --ldflags) --shared spammodule.c -o spammodule.so

Once again, you *will* need to adapt this to your platform and Python
distribution.
