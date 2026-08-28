:orphan:

.. _py_getconstant-full-list:

Constants for :c:func:`Py_GetConstant`
--------------------------------------

   .. stable-abi-note::

      All of the constant identifiers can be used when compiling for the
      :ref:`stable ABI <stable>`, since the version they were added in.

   Numeric values are only given for projects which cannot use the constant
   identifiers.

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

      - * .. rst-class:: omit-stable-abi-note
          .. c:macro:: Py_CONSTANT_Exc_ArithmeticError
        * 3.16
        * ``10``
        * :py:type:`ArithmeticError`
      - * .. rst-class:: omit-stable-abi-note
          .. c:macro:: Py_CONSTANT_Exc_AssertionError
        * 3.16
        * ``11``
        * :py:type:`AssertionError`
      - * .. rst-class:: omit-stable-abi-note
          .. c:macro:: Py_CONSTANT_Exc_AttributeError
        * 3.16
        * ``12``
        * :py:type:`AttributeError`
      - * .. rst-class:: omit-stable-abi-note
          .. c:macro:: Py_CONSTANT_Exc_BaseException
        * 3.16
        * ``13``
        * :py:type:`BaseException`
      - * .. rst-class:: omit-stable-abi-note
          .. c:macro:: Py_CONSTANT_Exc_BaseExceptionGroup
        * 3.16
        * ``14``
        * :py:type:`BaseExceptionGroup`
      - * .. rst-class:: omit-stable-abi-note
          .. c:macro:: Py_CONSTANT_Exc_BlockingIOError
        * 3.16
        * ``15``
        * :py:type:`BlockingIOError`
      - * .. rst-class:: omit-stable-abi-note
          .. c:macro:: Py_CONSTANT_Exc_BrokenPipeError
        * 3.16
        * ``16``
        * :py:type:`BrokenPipeError`
      - * .. rst-class:: omit-stable-abi-note
          .. c:macro:: Py_CONSTANT_Exc_BufferError
        * 3.16
        * ``17``
        * :py:type:`BufferError`
      - * .. rst-class:: omit-stable-abi-note
          .. c:macro:: Py_CONSTANT_Exc_BytesWarning
        * 3.16
        * ``18``
        * :py:type:`BytesWarning`
      - * .. rst-class:: omit-stable-abi-note
          .. c:macro:: Py_CONSTANT_Exc_ChildProcessError
        * 3.16
        * ``19``
        * :py:type:`ChildProcessError`
      - * .. rst-class:: omit-stable-abi-note
          .. c:macro:: Py_CONSTANT_Exc_ConnectionAbortedError
        * 3.16
        * ``20``
        * :py:type:`ConnectionAbortedError`
      - * .. rst-class:: omit-stable-abi-note
          .. c:macro:: Py_CONSTANT_Exc_ConnectionError
        * 3.16
        * ``21``
        * :py:type:`ConnectionError`
      - * .. rst-class:: omit-stable-abi-note
          .. c:macro:: Py_CONSTANT_Exc_ConnectionRefusedError
        * 3.16
        * ``22``
        * :py:type:`ConnectionRefusedError`
      - * .. rst-class:: omit-stable-abi-note
          .. c:macro:: Py_CONSTANT_Exc_ConnectionResetError
        * 3.16
        * ``23``
        * :py:type:`ConnectionResetError`
      - * .. rst-class:: omit-stable-abi-note
          .. c:macro:: Py_CONSTANT_Exc_DeprecationWarning
        * 3.16
        * ``24``
        * :py:type:`DeprecationWarning`
      - * .. rst-class:: omit-stable-abi-note
          .. c:macro:: Py_CONSTANT_Exc_EOFError
        * 3.16
        * ``25``
        * :py:type:`EOFError`
      - * .. rst-class:: omit-stable-abi-note
          .. c:macro:: Py_CONSTANT_Exc_EncodingWarning
        * 3.16
        * ``26``
        * :py:type:`EncodingWarning`
      - * .. rst-class:: omit-stable-abi-note
          .. c:macro:: Py_CONSTANT_Exc_Exception
        * 3.16
        * ``27``
        * :py:type:`Exception`
      - * .. rst-class:: omit-stable-abi-note
          .. c:macro:: Py_CONSTANT_Exc_FileExistsError
        * 3.16
        * ``28``
        * :py:type:`FileExistsError`
      - * .. rst-class:: omit-stable-abi-note
          .. c:macro:: Py_CONSTANT_Exc_FileNotFoundError
        * 3.16
        * ``29``
        * :py:type:`FileNotFoundError`
      - * .. rst-class:: omit-stable-abi-note
          .. c:macro:: Py_CONSTANT_Exc_FloatingPointError
        * 3.16
        * ``30``
        * :py:type:`FloatingPointError`
      - * .. rst-class:: omit-stable-abi-note
          .. c:macro:: Py_CONSTANT_Exc_FutureWarning
        * 3.16
        * ``31``
        * :py:type:`FutureWarning`
      - * .. rst-class:: omit-stable-abi-note
          .. c:macro:: Py_CONSTANT_Exc_GeneratorExit
        * 3.16
        * ``32``
        * :py:type:`GeneratorExit`
      - * .. rst-class:: omit-stable-abi-note
          .. c:macro:: Py_CONSTANT_Exc_ImportError
        * 3.16
        * ``33``
        * :py:type:`ImportError`
      - * .. rst-class:: omit-stable-abi-note
          .. c:macro:: Py_CONSTANT_Exc_ImportWarning
        * 3.16
        * ``34``
        * :py:type:`ImportWarning`
      - * .. rst-class:: omit-stable-abi-note
          .. c:macro:: Py_CONSTANT_Exc_IndentationError
        * 3.16
        * ``35``
        * :py:type:`IndentationError`
      - * .. rst-class:: omit-stable-abi-note
          .. c:macro:: Py_CONSTANT_Exc_IndexError
        * 3.16
        * ``36``
        * :py:type:`IndexError`
      - * .. rst-class:: omit-stable-abi-note
          .. c:macro:: Py_CONSTANT_Exc_InterruptedError
        * 3.16
        * ``37``
        * :py:type:`InterruptedError`
      - * .. rst-class:: omit-stable-abi-note
          .. c:macro:: Py_CONSTANT_Exc_IsADirectoryError
        * 3.16
        * ``38``
        * :py:type:`IsADirectoryError`
      - * .. rst-class:: omit-stable-abi-note
          .. c:macro:: Py_CONSTANT_Exc_KeyError
        * 3.16
        * ``39``
        * :py:type:`KeyError`
      - * .. rst-class:: omit-stable-abi-note
          .. c:macro:: Py_CONSTANT_Exc_KeyboardInterrupt
        * 3.16
        * ``40``
        * :py:type:`KeyboardInterrupt`
      - * .. rst-class:: omit-stable-abi-note
          .. c:macro:: Py_CONSTANT_Exc_LookupError
        * 3.16
        * ``41``
        * :py:type:`LookupError`
      - * .. rst-class:: omit-stable-abi-note
          .. c:macro:: Py_CONSTANT_Exc_MemoryError
        * 3.16
        * ``42``
        * :py:type:`MemoryError`
      - * .. rst-class:: omit-stable-abi-note
          .. c:macro:: Py_CONSTANT_Exc_ModuleNotFoundError
        * 3.16
        * ``43``
        * :py:type:`ModuleNotFoundError`
      - * .. rst-class:: omit-stable-abi-note
          .. c:macro:: Py_CONSTANT_Exc_NameError
        * 3.16
        * ``44``
        * :py:type:`NameError`
      - * .. rst-class:: omit-stable-abi-note
          .. c:macro:: Py_CONSTANT_Exc_NotADirectoryError
        * 3.16
        * ``45``
        * :py:type:`NotADirectoryError`
      - * .. rst-class:: omit-stable-abi-note
          .. c:macro:: Py_CONSTANT_Exc_NotImplementedError
        * 3.16
        * ``46``
        * :py:type:`NotImplementedError`
      - * .. rst-class:: omit-stable-abi-note
          .. c:macro:: Py_CONSTANT_Exc_OSError
        * 3.16
        * ``47``
        * :py:type:`OSError`
      - * .. rst-class:: omit-stable-abi-note
          .. c:macro:: Py_CONSTANT_Exc_OverflowError
        * 3.16
        * ``48``
        * :py:type:`OverflowError`
      - * .. rst-class:: omit-stable-abi-note
          .. c:macro:: Py_CONSTANT_Exc_PendingDeprecationWarning
        * 3.16
        * ``49``
        * :py:type:`PendingDeprecationWarning`
      - * .. rst-class:: omit-stable-abi-note
          .. c:macro:: Py_CONSTANT_Exc_PermissionError
        * 3.16
        * ``50``
        * :py:type:`PermissionError`
      - * .. rst-class:: omit-stable-abi-note
          .. c:macro:: Py_CONSTANT_Exc_ProcessLookupError
        * 3.16
        * ``51``
        * :py:type:`ProcessLookupError`
      - * .. rst-class:: omit-stable-abi-note
          .. c:macro:: Py_CONSTANT_Exc_RecursionError
        * 3.16
        * ``52``
        * :py:type:`RecursionError`
      - * .. rst-class:: omit-stable-abi-note
          .. c:macro:: Py_CONSTANT_Exc_ReferenceError
        * 3.16
        * ``53``
        * :py:type:`ReferenceError`
      - * .. rst-class:: omit-stable-abi-note
          .. c:macro:: Py_CONSTANT_Exc_ResourceWarning
        * 3.16
        * ``54``
        * :py:type:`ResourceWarning`
      - * .. rst-class:: omit-stable-abi-note
          .. c:macro:: Py_CONSTANT_Exc_RuntimeError
        * 3.16
        * ``55``
        * :py:type:`RuntimeError`
      - * .. rst-class:: omit-stable-abi-note
          .. c:macro:: Py_CONSTANT_Exc_RuntimeWarning
        * 3.16
        * ``56``
        * :py:type:`RuntimeWarning`
      - * .. rst-class:: omit-stable-abi-note
          .. c:macro:: Py_CONSTANT_Exc_StopAsyncIteration
        * 3.16
        * ``57``
        * :py:type:`StopAsyncIteration`
      - * .. rst-class:: omit-stable-abi-note
          .. c:macro:: Py_CONSTANT_Exc_StopIteration
        * 3.16
        * ``58``
        * :py:type:`StopIteration`
      - * .. rst-class:: omit-stable-abi-note
          .. c:macro:: Py_CONSTANT_Exc_SyntaxError
        * 3.16
        * ``59``
        * :py:type:`SyntaxError`
      - * .. rst-class:: omit-stable-abi-note
          .. c:macro:: Py_CONSTANT_Exc_SyntaxWarning
        * 3.16
        * ``60``
        * :py:type:`SyntaxWarning`
      - * .. rst-class:: omit-stable-abi-note
          .. c:macro:: Py_CONSTANT_Exc_SystemError
        * 3.16
        * ``61``
        * :py:type:`SystemError`
      - * .. rst-class:: omit-stable-abi-note
          .. c:macro:: Py_CONSTANT_Exc_SystemExit
        * 3.16
        * ``62``
        * :py:type:`SystemExit`
      - * .. rst-class:: omit-stable-abi-note
          .. c:macro:: Py_CONSTANT_Exc_TabError
        * 3.16
        * ``63``
        * :py:type:`TabError`
      - * .. rst-class:: omit-stable-abi-note
          .. c:macro:: Py_CONSTANT_Exc_TimeoutError
        * 3.16
        * ``64``
        * :py:type:`TimeoutError`
      - * .. rst-class:: omit-stable-abi-note
          .. c:macro:: Py_CONSTANT_Exc_TypeError
        * 3.16
        * ``65``
        * :py:type:`TypeError`
      - * .. rst-class:: omit-stable-abi-note
          .. c:macro:: Py_CONSTANT_Exc_UnboundLocalError
        * 3.16
        * ``66``
        * :py:type:`UnboundLocalError`
      - * .. rst-class:: omit-stable-abi-note
          .. c:macro:: Py_CONSTANT_Exc_UnicodeDecodeError
        * 3.16
        * ``67``
        * :py:type:`UnicodeDecodeError`
      - * .. rst-class:: omit-stable-abi-note
          .. c:macro:: Py_CONSTANT_Exc_UnicodeEncodeError
        * 3.16
        * ``68``
        * :py:type:`UnicodeEncodeError`
      - * .. rst-class:: omit-stable-abi-note
          .. c:macro:: Py_CONSTANT_Exc_UnicodeError
        * 3.16
        * ``69``
        * :py:type:`UnicodeError`
      - * .. rst-class:: omit-stable-abi-note
          .. c:macro:: Py_CONSTANT_Exc_UnicodeTranslateError
        * 3.16
        * ``70``
        * :py:type:`UnicodeTranslateError`
      - * .. rst-class:: omit-stable-abi-note
          .. c:macro:: Py_CONSTANT_Exc_UnicodeWarning
        * 3.16
        * ``71``
        * :py:type:`UnicodeWarning`
      - * .. rst-class:: omit-stable-abi-note
          .. c:macro:: Py_CONSTANT_Exc_UserWarning
        * 3.16
        * ``72``
        * :py:type:`UserWarning`
      - * .. rst-class:: omit-stable-abi-note
          .. c:macro:: Py_CONSTANT_Exc_ValueError
        * 3.16
        * ``73``
        * :py:type:`ValueError`
      - * .. rst-class:: omit-stable-abi-note
          .. c:macro:: Py_CONSTANT_Exc_Warning
        * 3.16
        * ``74``
        * :py:type:`Warning`
      - * .. rst-class:: omit-stable-abi-note
          .. c:macro:: Py_CONSTANT_Exc_ZeroDivisionError
        * 3.16
        * ``75``
        * :py:type:`ZeroDivisionError`
      - * .. rst-class:: omit-stable-abi-note
          .. c:macro:: Py_CONSTANT_BaseObject_Type
        * 3.16
        * ``76``
        * :py:class:`object`
      - * .. rst-class:: omit-stable-abi-note
          .. c:macro:: Py_CONSTANT_Bool_Type
        * 3.16
        * ``77``
        * :py:class:`bool`
      - * .. rst-class:: omit-stable-abi-note
          .. c:macro:: Py_CONSTANT_ByteArray_Type
        * 3.16
        * ``78``
        * :py:class:`bytearray`
      - * .. rst-class:: omit-stable-abi-note
          .. c:macro:: Py_CONSTANT_Bytes_Type
        * 3.16
        * ``79``
        * :py:class:`bytes`
      - * .. rst-class:: omit-stable-abi-note
          .. c:macro:: Py_CONSTANT_CFunction_Type
        * 3.16
        * ``80``
        * :py:class:`types.BuiltinFunctionType`
      - * .. rst-class:: omit-stable-abi-note
          .. c:macro:: Py_CONSTANT_Capsule_Type
        * 3.16
        * ``81``
        * :py:class:`types.CapsuleType`
      - * .. rst-class:: omit-stable-abi-note
          .. c:macro:: Py_CONSTANT_ClassMethodDescr_Type
        * 3.16
        * ``82``
        * :py:class:`types.ClassMethodDescriptorType`
      - * .. rst-class:: omit-stable-abi-note
          .. c:macro:: Py_CONSTANT_Complex_Type
        * 3.16
        * ``83``
        * :py:class:`complex`
      - * .. rst-class:: omit-stable-abi-note
          .. c:macro:: Py_CONSTANT_DictProxy_Type
        * 3.16
        * ``84``
        * :py:class:`types.MappingProxyType`
      - * .. rst-class:: omit-stable-abi-note
          .. c:macro:: Py_CONSTANT_Dict_Type
        * 3.16
        * ``85``
        * :py:class:`dict`
      - * .. rst-class:: omit-stable-abi-note
          .. c:macro:: Py_CONSTANT_Ellipsis_Type
        * 3.16
        * ``86``
        * :py:class:`types.EllipsisType`
      - * .. rst-class:: omit-stable-abi-note
          .. c:macro:: Py_CONSTANT_Enum_Type
        * 3.16
        * ``87``
        * :py:class:`enumerate`
      - * .. rst-class:: omit-stable-abi-note
          .. c:macro:: Py_CONSTANT_Filter_Type
        * 3.16
        * ``88``
        * :py:class:`filter`
      - * .. rst-class:: omit-stable-abi-note
          .. c:macro:: Py_CONSTANT_Float_Type
        * 3.16
        * ``89``
        * :py:class:`float`
      - * .. rst-class:: omit-stable-abi-note
          .. c:macro:: Py_CONSTANT_FrozenSet_Type
        * 3.16
        * ``90``
        * :py:class:`frozenset`
      - * .. rst-class:: omit-stable-abi-note
          .. c:macro:: Py_CONSTANT_GetSetDescr_Type
        * 3.16
        * ``91``
        * :py:class:`types.GetSetDescriptorType`
      - * .. rst-class:: omit-stable-abi-note
          .. c:macro:: Py_CONSTANT_List_Type
        * 3.16
        * ``92``
        * :py:class:`list`
      - * .. rst-class:: omit-stable-abi-note
          .. c:macro:: Py_CONSTANT_Long_Type
        * 3.16
        * ``93``
        * :py:class:`int`
      - * .. rst-class:: omit-stable-abi-note
          .. c:macro:: Py_CONSTANT_Map_Type
        * 3.16
        * ``94``
        * :py:class:`map`
      - * .. rst-class:: omit-stable-abi-note
          .. c:macro:: Py_CONSTANT_MemberDescr_Type
        * 3.16
        * ``95``
        * :py:class:`types.MemberDescriptorType`
      - * .. rst-class:: omit-stable-abi-note
          .. c:macro:: Py_CONSTANT_MemoryView_Type
        * 3.16
        * ``96``
        * :py:class:`memoryview`
      - * .. rst-class:: omit-stable-abi-note
          .. c:macro:: Py_CONSTANT_MethodDescr_Type
        * 3.16
        * ``97``
        * :py:class:`types.MethodDescriptorType`
      - * .. rst-class:: omit-stable-abi-note
          .. c:macro:: Py_CONSTANT_ModuleDef_Type
        * 3.16
        * ``98``
        * (:c:data:`PyModuleDef_Type`)
      - * .. rst-class:: omit-stable-abi-note
          .. c:macro:: Py_CONSTANT_Module_Type
        * 3.16
        * ``99``
        * :py:class:`types.ModuleType`
      - * .. rst-class:: omit-stable-abi-note
          .. c:macro:: Py_CONSTANT_Property_Type
        * 3.16
        * ``100``
        * :py:class:`property`
      - * .. rst-class:: omit-stable-abi-note
          .. c:macro:: Py_CONSTANT_Range_Type
        * 3.16
        * ``101``
        * :py:class:`range`
      - * .. rst-class:: omit-stable-abi-note
          .. c:macro:: Py_CONSTANT_Reversed_Type
        * 3.16
        * ``102``
        * :py:class:`reversed`
      - * .. rst-class:: omit-stable-abi-note
          .. c:macro:: Py_CONSTANT_Set_Type
        * 3.16
        * ``103``
        * :py:class:`set`
      - * .. rst-class:: omit-stable-abi-note
          .. c:macro:: Py_CONSTANT_Slice_Type
        * 3.16
        * ``104``
        * :py:class:`slice`
      - * .. rst-class:: omit-stable-abi-note
          .. c:macro:: Py_CONSTANT_Super_Type
        * 3.16
        * ``105``
        * :py:class:`super`
      - * .. rst-class:: omit-stable-abi-note
          .. c:macro:: Py_CONSTANT_TraceBack_Type
        * 3.16
        * ``106``
        * :py:class:`types.TracebackType`
      - * .. rst-class:: omit-stable-abi-note
          .. c:macro:: Py_CONSTANT_Tuple_Type
        * 3.16
        * ``107``
        * :py:class:`tuple`
      - * .. rst-class:: omit-stable-abi-note
          .. c:macro:: Py_CONSTANT_Type_Type
        * 3.16
        * ``108``
        * :py:class:`type`
      - * .. rst-class:: omit-stable-abi-note
          .. c:macro:: Py_CONSTANT_Unicode_Type
        * 3.16
        * ``109``
        * :py:class:`str`
      - * .. rst-class:: omit-stable-abi-note
          .. c:macro:: Py_CONSTANT_WrapperDescr_Type
        * 3.16
        * ``110``
        * :py:class:`types.WrapperDescriptorType`
      - * .. rst-class:: omit-stable-abi-note
          .. c:macro:: Py_CONSTANT_Zip_Type
        * 3.16
        * ``111``
        * :py:class:`zip`
      - * .. rst-class:: omit-stable-abi-note
          .. c:macro:: Py_CONSTANT_GenericAliasType
        * 3.16
        * ``112``
        * :py:class:`types.GenericAlias`
      - * .. rst-class:: omit-stable-abi-note
          .. c:macro:: Py_CONSTANT_Method_Type
        * 3.16
        * ``113``
        * :py:class:`types.MethodType`
