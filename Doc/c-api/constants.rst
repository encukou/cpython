:orphan:

.. _py_getconstant-full-list:

Constant identifiers for :c:func:`Py_GetConstant`
-------------------------------------------------

The following macros are defined for use as arguments to
:c:func:`Py_GetConstant`.

.. stable-abi-note::

   All of the constant identifiers can be used when compiling for the
   :ref:`stable ABI <stable>`, since the version they were added in.

Numeric values are only given for projects which cannot use the constant
identifiers.

.. omit-stable-abi-notes::

   .. list-table::
      :header-rows: 1

      - * Constant Identifier
        * Added in
        * Value
        * Returned object

      - * .. rst-class:: omit-stable-abi-note
          .. c:macro:: Py_CONSTANT_NONE
        * 3.13
        * ``0``
        * :py:data:`None`
      - * .. rst-class:: omit-stable-abi-note
          .. c:macro:: Py_CONSTANT_FALSE
        * 3.13
        * ``1``
        * :py:data:`False`
      - * .. rst-class:: omit-stable-abi-note
          .. c:macro:: Py_CONSTANT_TRUE
        * 3.13
        * ``2``
        * :py:data:`True`
      - * .. rst-class:: omit-stable-abi-note
          .. c:macro:: Py_CONSTANT_ELLIPSIS
        * 3.13
        * ``3``
        * :py:data:`Ellipsis`
      - * .. rst-class:: omit-stable-abi-note
          .. c:macro:: Py_CONSTANT_NOT_IMPLEMENTED
        * 3.13
        * ``4``
        * :py:data:`NotImplemented`
      - * .. rst-class:: omit-stable-abi-note
          .. c:macro:: Py_CONSTANT_ZERO
        * 3.13
        * ``5``
        * ``0``
      - * .. rst-class:: omit-stable-abi-note
          .. c:macro:: Py_CONSTANT_ONE
        * 3.13
        * ``6``
        * ``1``
      - * .. rst-class:: omit-stable-abi-note
          .. c:macro:: Py_CONSTANT_EMPTY_STR
        * 3.13
        * ``7``
        * ``''``
      - * .. rst-class:: omit-stable-abi-note
          .. c:macro:: Py_CONSTANT_EMPTY_BYTES
        * 3.13
        * ``8``
        * ``b''``
      - * .. rst-class:: omit-stable-abi-note
          .. c:macro:: Py_CONSTANT_EMPTY_TUPLE
        * 3.13
        * ``9``
        * ``()``

      - * .. c:macro:: Py_CONSTANT_Exc_ArithmeticError
        * 3.16
        * ``10``
        * :py:type:`ArithmeticError`
      - * .. c:macro:: Py_CONSTANT_Exc_AssertionError
        * 3.16
        * ``11``
        * :py:type:`AssertionError`
      - * .. c:macro:: Py_CONSTANT_Exc_AttributeError
        * 3.16
        * ``12``
        * :py:type:`AttributeError`
      - * .. c:macro:: Py_CONSTANT_Exc_BaseException
        * 3.16
        * ``13``
        * :py:type:`BaseException`
      - * .. c:macro:: Py_CONSTANT_Exc_BaseExceptionGroup
        * 3.16
        * ``14``
        * :py:type:`BaseExceptionGroup`
      - * .. c:macro:: Py_CONSTANT_Exc_BlockingIOError
        * 3.16
        * ``15``
        * :py:type:`BlockingIOError`
      - * .. c:macro:: Py_CONSTANT_Exc_BrokenPipeError
        * 3.16
        * ``16``
        * :py:type:`BrokenPipeError`
      - * .. c:macro:: Py_CONSTANT_Exc_BufferError
        * 3.16
        * ``17``
        * :py:type:`BufferError`
      - * .. c:macro:: Py_CONSTANT_Exc_BytesWarning
        * 3.16
        * ``18``
        * :py:type:`BytesWarning`
      - * .. c:macro:: Py_CONSTANT_Exc_ChildProcessError
        * 3.16
        * ``19``
        * :py:type:`ChildProcessError`
      - * .. c:macro:: Py_CONSTANT_Exc_ConnectionAbortedError
        * 3.16
        * ``20``
        * :py:type:`ConnectionAbortedError`
      - * .. c:macro:: Py_CONSTANT_Exc_ConnectionError
        * 3.16
        * ``21``
        * :py:type:`ConnectionError`
      - * .. c:macro:: Py_CONSTANT_Exc_ConnectionRefusedError
        * 3.16
        * ``22``
        * :py:type:`ConnectionRefusedError`
      - * .. c:macro:: Py_CONSTANT_Exc_ConnectionResetError
        * 3.16
        * ``23``
        * :py:type:`ConnectionResetError`
      - * .. c:macro:: Py_CONSTANT_Exc_DeprecationWarning
        * 3.16
        * ``24``
        * :py:type:`DeprecationWarning`
      - * .. c:macro:: Py_CONSTANT_Exc_EOFError
        * 3.16
        * ``25``
        * :py:type:`EOFError`
      - * .. c:macro:: Py_CONSTANT_Exc_EncodingWarning
        * 3.16
        * ``26``
        * :py:type:`EncodingWarning`
      - * .. c:macro:: Py_CONSTANT_Exc_Exception
        * 3.16
        * ``27``
        * :py:type:`Exception`
      - * .. c:macro:: Py_CONSTANT_Exc_FileExistsError
        * 3.16
        * ``28``
        * :py:type:`FileExistsError`
      - * .. c:macro:: Py_CONSTANT_Exc_FileNotFoundError
        * 3.16
        * ``29``
        * :py:type:`FileNotFoundError`
      - * .. c:macro:: Py_CONSTANT_Exc_FloatingPointError
        * 3.16
        * ``30``
        * :py:type:`FloatingPointError`
      - * .. c:macro:: Py_CONSTANT_Exc_FutureWarning
        * 3.16
        * ``31``
        * :py:type:`FutureWarning`
      - * .. c:macro:: Py_CONSTANT_Exc_GeneratorExit
        * 3.16
        * ``32``
        * :py:type:`GeneratorExit`
      - * .. c:macro:: Py_CONSTANT_Exc_ImportError
        * 3.16
        * ``33``
        * :py:type:`ImportError`
      - * .. c:macro:: Py_CONSTANT_Exc_ImportWarning
        * 3.16
        * ``34``
        * :py:type:`ImportWarning`
      - * .. c:macro:: Py_CONSTANT_Exc_IndentationError
        * 3.16
        * ``35``
        * :py:type:`IndentationError`
      - * .. c:macro:: Py_CONSTANT_Exc_IndexError
        * 3.16
        * ``36``
        * :py:type:`IndexError`
      - * .. c:macro:: Py_CONSTANT_Exc_InterruptedError
        * 3.16
        * ``37``
        * :py:type:`InterruptedError`
      - * .. c:macro:: Py_CONSTANT_Exc_IsADirectoryError
        * 3.16
        * ``38``
        * :py:type:`IsADirectoryError`
      - * .. c:macro:: Py_CONSTANT_Exc_KeyError
        * 3.16
        * ``39``
        * :py:type:`KeyError`
      - * .. c:macro:: Py_CONSTANT_Exc_KeyboardInterrupt
        * 3.16
        * ``40``
        * :py:type:`KeyboardInterrupt`
      - * .. c:macro:: Py_CONSTANT_Exc_LookupError
        * 3.16
        * ``41``
        * :py:type:`LookupError`
      - * .. c:macro:: Py_CONSTANT_Exc_MemoryError
        * 3.16
        * ``42``
        * :py:type:`MemoryError`
      - * .. c:macro:: Py_CONSTANT_Exc_ModuleNotFoundError
        * 3.16
        * ``43``
        * :py:type:`ModuleNotFoundError`
      - * .. c:macro:: Py_CONSTANT_Exc_NameError
        * 3.16
        * ``44``
        * :py:type:`NameError`
      - * .. c:macro:: Py_CONSTANT_Exc_NotADirectoryError
        * 3.16
        * ``45``
        * :py:type:`NotADirectoryError`
      - * .. c:macro:: Py_CONSTANT_Exc_NotImplementedError
        * 3.16
        * ``46``
        * :py:type:`NotImplementedError`
      - * .. c:macro:: Py_CONSTANT_Exc_OSError
        * 3.16
        * ``47``
        * :py:type:`OSError`
      - * .. c:macro:: Py_CONSTANT_Exc_OverflowError
        * 3.16
        * ``48``
        * :py:type:`OverflowError`
      - * .. c:macro:: Py_CONSTANT_Exc_PendingDeprecationWarning
        * 3.16
        * ``49``
        * :py:type:`PendingDeprecationWarning`
      - * .. c:macro:: Py_CONSTANT_Exc_PermissionError
        * 3.16
        * ``50``
        * :py:type:`PermissionError`
      - * .. c:macro:: Py_CONSTANT_Exc_ProcessLookupError
        * 3.16
        * ``51``
        * :py:type:`ProcessLookupError`
      - * .. c:macro:: Py_CONSTANT_Exc_RecursionError
        * 3.16
        * ``52``
        * :py:type:`RecursionError`
      - * .. c:macro:: Py_CONSTANT_Exc_ReferenceError
        * 3.16
        * ``53``
        * :py:type:`ReferenceError`
      - * .. c:macro:: Py_CONSTANT_Exc_ResourceWarning
        * 3.16
        * ``54``
        * :py:type:`ResourceWarning`
      - * .. c:macro:: Py_CONSTANT_Exc_RuntimeError
        * 3.16
        * ``55``
        * :py:type:`RuntimeError`
      - * .. c:macro:: Py_CONSTANT_Exc_RuntimeWarning
        * 3.16
        * ``56``
        * :py:type:`RuntimeWarning`
      - * .. c:macro:: Py_CONSTANT_Exc_StopAsyncIteration
        * 3.16
        * ``57``
        * :py:type:`StopAsyncIteration`
      - * .. c:macro:: Py_CONSTANT_Exc_StopIteration
        * 3.16
        * ``58``
        * :py:type:`StopIteration`
      - * .. c:macro:: Py_CONSTANT_Exc_SyntaxError
        * 3.16
        * ``59``
        * :py:type:`SyntaxError`
      - * .. c:macro:: Py_CONSTANT_Exc_SyntaxWarning
        * 3.16
        * ``60``
        * :py:type:`SyntaxWarning`
      - * .. c:macro:: Py_CONSTANT_Exc_SystemError
        * 3.16
        * ``61``
        * :py:type:`SystemError`
      - * .. c:macro:: Py_CONSTANT_Exc_SystemExit
        * 3.16
        * ``62``
        * :py:type:`SystemExit`
      - * .. c:macro:: Py_CONSTANT_Exc_TabError
        * 3.16
        * ``63``
        * :py:type:`TabError`
      - * .. c:macro:: Py_CONSTANT_Exc_TimeoutError
        * 3.16
        * ``64``
        * :py:type:`TimeoutError`
      - * .. c:macro:: Py_CONSTANT_Exc_TypeError
        * 3.16
        * ``65``
        * :py:type:`TypeError`
      - * .. c:macro:: Py_CONSTANT_Exc_UnboundLocalError
        * 3.16
        * ``66``
        * :py:type:`UnboundLocalError`
      - * .. c:macro:: Py_CONSTANT_Exc_UnicodeDecodeError
        * 3.16
        * ``67``
        * :py:type:`UnicodeDecodeError`
      - * .. c:macro:: Py_CONSTANT_Exc_UnicodeEncodeError
        * 3.16
        * ``68``
        * :py:type:`UnicodeEncodeError`
      - * .. c:macro:: Py_CONSTANT_Exc_UnicodeError
        * 3.16
        * ``69``
        * :py:type:`UnicodeError`
      - * .. c:macro:: Py_CONSTANT_Exc_UnicodeTranslateError
        * 3.16
        * ``70``
        * :py:type:`UnicodeTranslateError`
      - * .. c:macro:: Py_CONSTANT_Exc_UnicodeWarning
        * 3.16
        * ``71``
        * :py:type:`UnicodeWarning`
      - * .. c:macro:: Py_CONSTANT_Exc_UserWarning
        * 3.16
        * ``72``
        * :py:type:`UserWarning`
      - * .. c:macro:: Py_CONSTANT_Exc_ValueError
        * 3.16
        * ``73``
        * :py:type:`ValueError`
      - * .. c:macro:: Py_CONSTANT_Exc_Warning
        * 3.16
        * ``74``
        * :py:type:`Warning`
      - * .. c:macro:: Py_CONSTANT_Exc_ZeroDivisionError
        * 3.16
        * ``75``
        * :py:type:`ZeroDivisionError`
      - * .. c:macro:: Py_CONSTANT_BaseObject_Type
        * 3.16
        * ``76``
        * :c:data:`PyBaseObject_Type`
      - * .. c:macro:: Py_CONSTANT_Bool_Type
        * 3.16
        * ``77``
        * :c:data:`PyBool_Type`
      - * .. c:macro:: Py_CONSTANT_ByteArray_Type
        * 3.16
        * ``78``
        * :c:data:`PyByteArray_Type`
      - * .. c:macro:: Py_CONSTANT_Bytes_Type
        * 3.16
        * ``79``
        * :c:data:`PyBytes_Type`
      - * .. c:macro:: Py_CONSTANT_CFunction_Type
        * 3.16
        * ``80``
        * :c:data:`PyCFunction_Type`
      - * .. c:macro:: Py_CONSTANT_Capsule_Type
        * 3.16
        * ``81``
        * :c:data:`PyCapsule_Type`
      - * .. c:macro:: Py_CONSTANT_ClassMethodDescr_Type
        * 3.16
        * ``82``
        * :c:data:`PyClassMethodDescr_Type`
      - * .. c:macro:: Py_CONSTANT_Complex_Type
        * 3.16
        * ``83``
        * :c:data:`PyComplex_Type`
      - * .. c:macro:: Py_CONSTANT_DictProxy_Type
        * 3.16
        * ``84``
        * :c:data:`PyDictProxy_Type`
      - * .. c:macro:: Py_CONSTANT_Dict_Type
        * 3.16
        * ``85``
        * :c:data:`PyDict_Type`
      - * .. c:macro:: Py_CONSTANT_Ellipsis_Type
        * 3.16
        * ``86``
        * :c:data:`PyEllipsis_Type`
      - * .. c:macro:: Py_CONSTANT_Enum_Type
        * 3.16
        * ``87``
        * :c:data:`PyEnum_Type`
      - * .. c:macro:: Py_CONSTANT_Filter_Type
        * 3.16
        * ``88``
        * :c:data:`PyFilter_Type`
      - * .. c:macro:: Py_CONSTANT_Float_Type
        * 3.16
        * ``89``
        * :c:data:`PyFloat_Type`
      - * .. c:macro:: Py_CONSTANT_FrozenSet_Type
        * 3.16
        * ``90``
        * :c:data:`PyFrozenSet_Type`
      - * .. c:macro:: Py_CONSTANT_GetSetDescr_Type
        * 3.16
        * ``91``
        * :c:data:`PyGetSetDescr_Type`
      - * .. c:macro:: Py_CONSTANT_List_Type
        * 3.16
        * ``92``
        * :c:data:`PyList_Type`
      - * .. c:macro:: Py_CONSTANT_Long_Type
        * 3.16
        * ``93``
        * :c:data:`PyLong_Type`
      - * .. c:macro:: Py_CONSTANT_Map_Type
        * 3.16
        * ``94``
        * :c:data:`PyMap_Type`
      - * .. c:macro:: Py_CONSTANT_MemberDescr_Type
        * 3.16
        * ``95``
        * :c:data:`PyMemberDescr_Type`
      - * .. c:macro:: Py_CONSTANT_MemoryView_Type
        * 3.16
        * ``96``
        * :c:data:`PyMemoryView_Type`
      - * .. c:macro:: Py_CONSTANT_MethodDescr_Type
        * 3.16
        * ``97``
        * :c:data:`PyMethodDescr_Type`
      - * .. c:macro:: Py_CONSTANT_Module_Type
        * 3.16
        * ``98``
        * :c:data:`PyModule_Type`
      - * .. c:macro:: Py_CONSTANT_Property_Type
        * 3.16
        * ``99``
        * :c:data:`PyProperty_Type`
      - * .. c:macro:: Py_CONSTANT_Range_Type
        * 3.16
        * ``100``
        * :c:data:`PyRange_Type`
      - * .. c:macro:: Py_CONSTANT_Reversed_Type
        * 3.16
        * ``101``
        * :c:data:`PyReversed_Type`
      - * .. c:macro:: Py_CONSTANT_Set_Type
        * 3.16
        * ``102``
        * :c:data:`PySet_Type`
      - * .. c:macro:: Py_CONSTANT_Slice_Type
        * 3.16
        * ``103``
        * :c:data:`PySlice_Type`
      - * .. c:macro:: Py_CONSTANT_Super_Type
        * 3.16
        * ``104``
        * :c:data:`PySuper_Type`
      - * .. c:macro:: Py_CONSTANT_TraceBack_Type
        * 3.16
        * ``105``
        * :c:data:`PyTraceBack_Type`
      - * .. c:macro:: Py_CONSTANT_Tuple_Type
        * 3.16
        * ``106``
        * :c:data:`PyTuple_Type`
      - * .. c:macro:: Py_CONSTANT_Type_Type
        * 3.16
        * ``107``
        * :c:data:`PyType_Type`
      - * .. c:macro:: Py_CONSTANT_Unicode_Type
        * 3.16
        * ``108``
        * :c:data:`PyUnicode_Type`
      - * .. c:macro:: Py_CONSTANT_WrapperDescr_Type
        * 3.16
        * ``109``
        * :c:data:`PyWrapperDescr_Type`
      - * .. c:macro:: Py_CONSTANT_Zip_Type
        * 3.16
        * ``110``
        * :c:data:`PyZip_Type`
      - * .. c:macro:: Py_CONSTANT_GenericAliasType
        * 3.16
        * ``111``
        * :c:data:`Py_GenericAliasType`
      - * .. c:macro:: Py_CONSTANT_Method_Type
        * 3.16
        * ``112``
        * :c:data:`PyMethod_Type`
