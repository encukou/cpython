    {"__getattribute__", offsetof(PyTypeObject, tp_getattr), (void *)(NULL), NULL, PyDoc_STR(""), .name_strobj = &_Py_ID(__getattribute__)},
    {"__getattr__", offsetof(PyTypeObject, tp_getattr), (void *)(NULL), NULL, PyDoc_STR(""), .name_strobj = &_Py_ID(__getattr__)},
    {"__setattr__", offsetof(PyTypeObject, tp_setattr), (void *)(NULL), NULL, PyDoc_STR(""), .name_strobj = &_Py_ID(__setattr__)},
    {"__delattr__", offsetof(PyTypeObject, tp_setattr), (void *)(NULL), NULL, PyDoc_STR(""), .name_strobj = &_Py_ID(__delattr__)},
    {"__repr__", offsetof(PyTypeObject, tp_repr), (void *)(slot_tp_repr), wrap_unaryfunc, PyDoc_STR("__repr__($self, /)\n--\n\nReturn repr(self)."), .name_strobj = &_Py_ID(__repr__)}
                                                          ,
    {"__hash__", offsetof(PyTypeObject, tp_hash), (void *)(slot_tp_hash), wrap_hashfunc, PyDoc_STR("__hash__($self, /)\n--\n\nReturn hash(self)."), .name_strobj = &_Py_ID(__hash__)}
                                                          ,
    {"__call__", offsetof(PyTypeObject, tp_call), (void *)(slot_tp_call), (wrapperfunc)(void(*)(void))wrap_call, PyDoc_STR("__call__($self, /, *args, **kwargs)\n--\n\nCall self as a function."), PyWrapperFlag_KEYWORDS, .name_strobj = &_Py_ID(__call__) }

                                  ,
    {"__str__", offsetof(PyTypeObject, tp_str), (void *)(slot_tp_str), wrap_unaryfunc, PyDoc_STR("__str__($self, /)\n--\n\nReturn str(self)."), .name_strobj = &_Py_ID(__str__)}
                                                        ,
    {"__getattribute__", offsetof(PyTypeObject, tp_getattro), (void *)(_Py_slot_tp_getattr_hook), wrap_binaryfunc, PyDoc_STR("__getattribute__($self, name, /)\n--\n\nReturn getattr(self, name)."), .name_strobj = &_Py_ID(__getattribute__)}

                                                                                 ,
    {"__getattr__", offsetof(PyTypeObject, tp_getattro), (void *)(_Py_slot_tp_getattr_hook), NULL, PyDoc_STR("__getattr__($self, name, /)\n--\n\nImplement getattr(self, name)."), .name_strobj = &_Py_ID(__getattr__)}
                                                                               ,
    {"__setattr__", offsetof(PyTypeObject, tp_setattro), (void *)(slot_tp_setattro), wrap_setattr, PyDoc_STR("__setattr__($self, name, value, /)\n--\n\nImplement setattr(self, name, value)."), .name_strobj = &_Py_ID(__setattr__)}
                                                                                             ,
    {"__delattr__", offsetof(PyTypeObject, tp_setattro), (void *)(slot_tp_setattro), wrap_delattr, PyDoc_STR("__delattr__($self, name, /)\n--\n\nImplement delattr(self, name)."), .name_strobj = &_Py_ID(__delattr__)}
                                                                               ,
    {"__lt__", offsetof(PyTypeObject, tp_richcompare), (void *)(slot_tp_richcompare), richcmp_lt, PyDoc_STR("__lt__($self, value, /)\n--\n\nReturn self<value."), .name_strobj = &_Py_ID(__lt__)}
                                                               ,
    {"__le__", offsetof(PyTypeObject, tp_richcompare), (void *)(slot_tp_richcompare), richcmp_le, PyDoc_STR("__le__($self, value, /)\n--\n\nReturn self<=value."), .name_strobj = &_Py_ID(__le__)}
                                                                ,
    {"__eq__", offsetof(PyTypeObject, tp_richcompare), (void *)(slot_tp_richcompare), richcmp_eq, PyDoc_STR("__eq__($self, value, /)\n--\n\nReturn self==value."), .name_strobj = &_Py_ID(__eq__)}
                                                                ,
    {"__ne__", offsetof(PyTypeObject, tp_richcompare), (void *)(slot_tp_richcompare), richcmp_ne, PyDoc_STR("__ne__($self, value, /)\n--\n\nReturn self!=value."), .name_strobj = &_Py_ID(__ne__)}
                                                                ,
    {"__gt__", offsetof(PyTypeObject, tp_richcompare), (void *)(slot_tp_richcompare), richcmp_gt, PyDoc_STR("__gt__($self, value, /)\n--\n\nReturn self>value."), .name_strobj = &_Py_ID(__gt__)}
                                                               ,
    {"__ge__", offsetof(PyTypeObject, tp_richcompare), (void *)(slot_tp_richcompare), richcmp_ge, PyDoc_STR("__ge__($self, value, /)\n--\n\nReturn self>=value."), .name_strobj = &_Py_ID(__ge__)}
                                                                ,
    {"__iter__", offsetof(PyTypeObject, tp_iter), (void *)(slot_tp_iter), wrap_unaryfunc, PyDoc_STR("__iter__($self, /)\n--\n\nImplement iter(self)."), .name_strobj = &_Py_ID(__iter__)}
                                                             ,
    {"__next__", offsetof(PyTypeObject, tp_iternext), (void *)(slot_tp_iternext), wrap_next, PyDoc_STR("__next__($self, /)\n--\n\nImplement next(self)."), .name_strobj = &_Py_ID(__next__)}
                                                             ,
    {"__get__", offsetof(PyTypeObject, tp_descr_get), (void *)(slot_tp_descr_get), wrap_descr_get, PyDoc_STR("__get__($self, instance, owner=None, /)\n--\n\nReturn an attribute of instance, which is of type owner."), .name_strobj = &_Py_ID(__get__)}
                                                                                                                     ,
    {"__set__", offsetof(PyTypeObject, tp_descr_set), (void *)(slot_tp_descr_set), wrap_descr_set, PyDoc_STR("__set__($self, instance, value, /)\n--\n\nSet an attribute of instance to value."), .name_strobj = &_Py_ID(__set__)}
                                                                                              ,
    {"__delete__", offsetof(PyTypeObject, tp_descr_set), (void *)(slot_tp_descr_set), wrap_descr_delete, PyDoc_STR("__delete__($self, instance, /)\n--\n\nDelete an attribute of instance."), .name_strobj = &_Py_ID(__delete__)}

                                                                                    ,
    {"__init__", offsetof(PyTypeObject, tp_init), (void *)(slot_tp_init), (wrapperfunc)(void(*)(void))wrap_init, PyDoc_STR("__init__($self, /, *args, **kwargs)\n--\n\n" "Initialize self.  See help(type(self)) for accurate signature."), PyWrapperFlag_KEYWORDS, .name_strobj = &_Py_ID(__init__) }


                                  ,
    {"__new__", offsetof(PyTypeObject, tp_new), (void *)(slot_tp_new), NULL, PyDoc_STR("__new__(type, /, *args, **kwargs)\n--\n\n" "Create and return new object.  See help(type) for accurate signature."), .name_strobj = &_Py_ID(__new__)}

                                                                                   ,
    {"__del__", offsetof(PyTypeObject, tp_finalize), (void *)(slot_tp_finalize), (wrapperfunc)wrap_del, PyDoc_STR("__del__($self, /)\n--\n\n" "Called when the instance is about to be destroyed."), .name_strobj = &_Py_ID(__del__)}

                                                                ,

    {"__buffer__", offsetof(PyHeapTypeObject, as_buffer.bf_getbuffer), (void *)(slot_bf_getbuffer), wrap_buffer, PyDoc_STR("__buffer__($self, flags, /)\n--\n\n" "Return a buffer object that exposes the underlying memory of the object."), .name_strobj = &_Py_ID(__buffer__) }

                                                                                       ,
    {"__release_buffer__", offsetof(PyHeapTypeObject, as_buffer.bf_releasebuffer), (void *)(slot_bf_releasebuffer), wrap_releasebuffer, PyDoc_STR("__release_buffer__($self, buffer, /)\n--\n\n" "Release the buffer object that exposes the underlying memory of the object."), .name_strobj = &_Py_ID(__release_buffer__) }

                                                                                          ,

    {"__await__", offsetof(PyHeapTypeObject, as_async.am_await), (void *)(slot_am_await), wrap_unaryfunc, PyDoc_STR("__await__($self, /)\n--\n\nReturn an iterator to be used in await expression."), .name_strobj = &_Py_ID(__await__) }
                                                                                           ,
    {"__aiter__", offsetof(PyHeapTypeObject, as_async.am_aiter), (void *)(slot_am_aiter), wrap_unaryfunc, PyDoc_STR("__aiter__($self, /)\n--\n\nReturn an awaitable, that resolves in asynchronous iterator."), .name_strobj = &_Py_ID(__aiter__) }
                                                                                                     ,
    {"__anext__", offsetof(PyHeapTypeObject, as_async.am_anext), (void *)(slot_am_anext), wrap_unaryfunc, PyDoc_STR("__anext__($self, /)\n--\n\nReturn a value or raise StopAsyncIteration."), .name_strobj = &_Py_ID(__anext__) }
                                                                                    ,

    {"__add__", offsetof(PyHeapTypeObject, as_number.nb_add), (void *)(slot_nb_add), wrap_binaryfunc_l, PyDoc_STR("__add__" "($self, value, /)\n--\n\nReturn self" "+" "value."), .name_strobj = &_Py_ID(__add__) }
               ,
    {"__radd__", offsetof(PyHeapTypeObject, as_number.nb_add), (void *)(slot_nb_add), wrap_binaryfunc_r, PyDoc_STR("__radd__" "($self, value, /)\n--\n\nReturn value" "+" "self."), .name_strobj = &_Py_ID(__radd__) }
               ,
    {"__sub__", offsetof(PyHeapTypeObject, as_number.nb_subtract), (void *)(slot_nb_subtract), wrap_binaryfunc_l, PyDoc_STR("__sub__" "($self, value, /)\n--\n\nReturn self" "-" "value."), .name_strobj = &_Py_ID(__sub__) }
               ,
    {"__rsub__", offsetof(PyHeapTypeObject, as_number.nb_subtract), (void *)(slot_nb_subtract), wrap_binaryfunc_r, PyDoc_STR("__rsub__" "($self, value, /)\n--\n\nReturn value" "-" "self."), .name_strobj = &_Py_ID(__rsub__) }
               ,
    {"__mul__", offsetof(PyHeapTypeObject, as_number.nb_multiply), (void *)(slot_nb_multiply), wrap_binaryfunc_l, PyDoc_STR("__mul__" "($self, value, /)\n--\n\nReturn self" "*" "value."), .name_strobj = &_Py_ID(__mul__) }
               ,
    {"__rmul__", offsetof(PyHeapTypeObject, as_number.nb_multiply), (void *)(slot_nb_multiply), wrap_binaryfunc_r, PyDoc_STR("__rmul__" "($self, value, /)\n--\n\nReturn value" "*" "self."), .name_strobj = &_Py_ID(__rmul__) }
               ,
    {"__mod__", offsetof(PyHeapTypeObject, as_number.nb_remainder), (void *)(slot_nb_remainder), wrap_binaryfunc_l, PyDoc_STR("__mod__" "($self, value, /)\n--\n\nReturn self" "%" "value."), .name_strobj = &_Py_ID(__mod__) }
               ,
    {"__rmod__", offsetof(PyHeapTypeObject, as_number.nb_remainder), (void *)(slot_nb_remainder), wrap_binaryfunc_r, PyDoc_STR("__rmod__" "($self, value, /)\n--\n\nReturn value" "%" "self."), .name_strobj = &_Py_ID(__rmod__) }
               ,
    {"__divmod__", offsetof(PyHeapTypeObject, as_number.nb_divmod), (void *)(slot_nb_divmod), wrap_binaryfunc_l, PyDoc_STR("__divmod__" "($self, value, /)\n--\n\n" "Return divmod(self, value)."), .name_strobj = &_Py_ID(__divmod__) }
                                         ,
    {"__rdivmod__", offsetof(PyHeapTypeObject, as_number.nb_divmod), (void *)(slot_nb_divmod), wrap_binaryfunc_r, PyDoc_STR("__rdivmod__" "($self, value, /)\n--\n\n" "Return divmod(value, self)."), .name_strobj = &_Py_ID(__rdivmod__) }
                                         ,
    {"__pow__", offsetof(PyHeapTypeObject, as_number.nb_power), (void *)(slot_nb_power), wrap_ternaryfunc, PyDoc_STR("__pow__($self, value, mod=None, /)\n--\n\nReturn pow(self, value, mod)."), .name_strobj = &_Py_ID(__pow__) }
                                                                                     ,
    {"__rpow__", offsetof(PyHeapTypeObject, as_number.nb_power), (void *)(slot_nb_power), wrap_ternaryfunc_r, PyDoc_STR("__rpow__($self, value, mod=None, /)\n--\n\nReturn pow(value, self, mod)."), .name_strobj = &_Py_ID(__rpow__) }
                                                                                      ,
    {"__neg__", offsetof(PyHeapTypeObject, as_number.nb_negative), (void *)(slot_nb_negative), wrap_unaryfunc, PyDoc_STR("__neg__" "($self, /)\n--\n\n" "-self"), .name_strobj = &_Py_ID(__neg__) },
    {"__pos__", offsetof(PyHeapTypeObject, as_number.nb_positive), (void *)(slot_nb_positive), wrap_unaryfunc, PyDoc_STR("__pos__" "($self, /)\n--\n\n" "+self"), .name_strobj = &_Py_ID(__pos__) },
    {"__abs__", offsetof(PyHeapTypeObject, as_number.nb_absolute), (void *)(slot_nb_absolute), wrap_unaryfunc, PyDoc_STR("__abs__" "($self, /)\n--\n\n" "abs(self)"), .name_strobj = &_Py_ID(__abs__) }
                       ,
    {"__bool__", offsetof(PyHeapTypeObject, as_number.nb_bool), (void *)(slot_nb_bool), wrap_inquirypred, PyDoc_STR("__bool__" "($self, /)\n--\n\n" "True if self else False"), .name_strobj = &_Py_ID(__bool__) }
                                     ,
    {"__invert__", offsetof(PyHeapTypeObject, as_number.nb_invert), (void *)(slot_nb_invert), wrap_unaryfunc, PyDoc_STR("__invert__" "($self, /)\n--\n\n" "~self"), .name_strobj = &_Py_ID(__invert__) },
    {"__lshift__", offsetof(PyHeapTypeObject, as_number.nb_lshift), (void *)(slot_nb_lshift), wrap_binaryfunc_l, PyDoc_STR("__lshift__" "($self, value, /)\n--\n\nReturn self" "<<" "value."), .name_strobj = &_Py_ID(__lshift__) },
    {"__rlshift__", offsetof(PyHeapTypeObject, as_number.nb_lshift), (void *)(slot_nb_lshift), wrap_binaryfunc_r, PyDoc_STR("__rlshift__" "($self, value, /)\n--\n\nReturn value" "<<" "self."), .name_strobj = &_Py_ID(__rlshift__) },
    {"__rshift__", offsetof(PyHeapTypeObject, as_number.nb_rshift), (void *)(slot_nb_rshift), wrap_binaryfunc_l, PyDoc_STR("__rshift__" "($self, value, /)\n--\n\nReturn self" ">>" "value."), .name_strobj = &_Py_ID(__rshift__) },
    {"__rrshift__", offsetof(PyHeapTypeObject, as_number.nb_rshift), (void *)(slot_nb_rshift), wrap_binaryfunc_r, PyDoc_STR("__rrshift__" "($self, value, /)\n--\n\nReturn value" ">>" "self."), .name_strobj = &_Py_ID(__rrshift__) },
    {"__and__", offsetof(PyHeapTypeObject, as_number.nb_and), (void *)(slot_nb_and), wrap_binaryfunc_l, PyDoc_STR("__and__" "($self, value, /)\n--\n\nReturn self" "&" "value."), .name_strobj = &_Py_ID(__and__) },
    {"__rand__", offsetof(PyHeapTypeObject, as_number.nb_and), (void *)(slot_nb_and), wrap_binaryfunc_r, PyDoc_STR("__rand__" "($self, value, /)\n--\n\nReturn value" "&" "self."), .name_strobj = &_Py_ID(__rand__) },
    {"__xor__", offsetof(PyHeapTypeObject, as_number.nb_xor), (void *)(slot_nb_xor), wrap_binaryfunc_l, PyDoc_STR("__xor__" "($self, value, /)\n--\n\nReturn self" "^" "value."), .name_strobj = &_Py_ID(__xor__) },
    {"__rxor__", offsetof(PyHeapTypeObject, as_number.nb_xor), (void *)(slot_nb_xor), wrap_binaryfunc_r, PyDoc_STR("__rxor__" "($self, value, /)\n--\n\nReturn value" "^" "self."), .name_strobj = &_Py_ID(__rxor__) },
    {"__or__", offsetof(PyHeapTypeObject, as_number.nb_or), (void *)(slot_nb_or), wrap_binaryfunc_l, PyDoc_STR("__or__" "($self, value, /)\n--\n\nReturn self" "|" "value."), .name_strobj = &_Py_ID(__or__) },
    {"__ror__", offsetof(PyHeapTypeObject, as_number.nb_or), (void *)(slot_nb_or), wrap_binaryfunc_r, PyDoc_STR("__ror__" "($self, value, /)\n--\n\nReturn value" "|" "self."), .name_strobj = &_Py_ID(__ror__) },
    {"__int__", offsetof(PyHeapTypeObject, as_number.nb_int), (void *)(slot_nb_int), wrap_unaryfunc, PyDoc_STR("__int__" "($self, /)\n--\n\n" "int(self)"), .name_strobj = &_Py_ID(__int__) }
                       ,
    {"__float__", offsetof(PyHeapTypeObject, as_number.nb_float), (void *)(slot_nb_float), wrap_unaryfunc, PyDoc_STR("__float__" "($self, /)\n--\n\n" "float(self)"), .name_strobj = &_Py_ID(__float__) }
                         ,
    {"__iadd__", offsetof(PyHeapTypeObject, as_number.nb_inplace_add), (void *)(slot_nb_inplace_add), wrap_binaryfunc, PyDoc_STR("__iadd__" "($self, value, /)\n--\n\nReturn self" "+=" "value."), .name_strobj = &_Py_ID(__iadd__) }
                                 ,
    {"__isub__", offsetof(PyHeapTypeObject, as_number.nb_inplace_subtract), (void *)(slot_nb_inplace_subtract), wrap_binaryfunc, PyDoc_STR("__isub__" "($self, value, /)\n--\n\nReturn self" "-=" "value."), .name_strobj = &_Py_ID(__isub__) }
                                 ,
    {"__imul__", offsetof(PyHeapTypeObject, as_number.nb_inplace_multiply), (void *)(slot_nb_inplace_multiply), wrap_binaryfunc, PyDoc_STR("__imul__" "($self, value, /)\n--\n\nReturn self" "*=" "value."), .name_strobj = &_Py_ID(__imul__) }
                                 ,
    {"__imod__", offsetof(PyHeapTypeObject, as_number.nb_inplace_remainder), (void *)(slot_nb_inplace_remainder), wrap_binaryfunc, PyDoc_STR("__imod__" "($self, value, /)\n--\n\nReturn self" "%=" "value."), .name_strobj = &_Py_ID(__imod__) }
                                 ,
    {"__ipow__", offsetof(PyHeapTypeObject, as_number.nb_inplace_power), (void *)(slot_nb_inplace_power), wrap_ternaryfunc, PyDoc_STR("__ipow__" "($self, value, /)\n--\n\nReturn self" "**=" "value."), .name_strobj = &_Py_ID(__ipow__) }
                                   ,
    {"__ilshift__", offsetof(PyHeapTypeObject, as_number.nb_inplace_lshift), (void *)(slot_nb_inplace_lshift), wrap_binaryfunc, PyDoc_STR("__ilshift__" "($self, value, /)\n--\n\nReturn self" "<<=" "value."), .name_strobj = &_Py_ID(__ilshift__) }
                                  ,
    {"__irshift__", offsetof(PyHeapTypeObject, as_number.nb_inplace_rshift), (void *)(slot_nb_inplace_rshift), wrap_binaryfunc, PyDoc_STR("__irshift__" "($self, value, /)\n--\n\nReturn self" ">>=" "value."), .name_strobj = &_Py_ID(__irshift__) }
                                  ,
    {"__iand__", offsetof(PyHeapTypeObject, as_number.nb_inplace_and), (void *)(slot_nb_inplace_and), wrap_binaryfunc, PyDoc_STR("__iand__" "($self, value, /)\n--\n\nReturn self" "&=" "value."), .name_strobj = &_Py_ID(__iand__) }
                                 ,
    {"__ixor__", offsetof(PyHeapTypeObject, as_number.nb_inplace_xor), (void *)(slot_nb_inplace_xor), wrap_binaryfunc, PyDoc_STR("__ixor__" "($self, value, /)\n--\n\nReturn self" "^=" "value."), .name_strobj = &_Py_ID(__ixor__) }
                                 ,
    {"__ior__", offsetof(PyHeapTypeObject, as_number.nb_inplace_or), (void *)(slot_nb_inplace_or), wrap_binaryfunc, PyDoc_STR("__ior__" "($self, value, /)\n--\n\nReturn self" "|=" "value."), .name_strobj = &_Py_ID(__ior__) }
                                 ,
    {"__floordiv__", offsetof(PyHeapTypeObject, as_number.nb_floor_divide), (void *)(slot_nb_floor_divide), wrap_binaryfunc_l, PyDoc_STR("__floordiv__" "($self, value, /)\n--\n\nReturn self" "//" "value."), .name_strobj = &_Py_ID(__floordiv__) },
    {"__rfloordiv__", offsetof(PyHeapTypeObject, as_number.nb_floor_divide), (void *)(slot_nb_floor_divide), wrap_binaryfunc_r, PyDoc_STR("__rfloordiv__" "($self, value, /)\n--\n\nReturn value" "//" "self."), .name_strobj = &_Py_ID(__rfloordiv__) },
    {"__truediv__", offsetof(PyHeapTypeObject, as_number.nb_true_divide), (void *)(slot_nb_true_divide), wrap_binaryfunc_l, PyDoc_STR("__truediv__" "($self, value, /)\n--\n\nReturn self" "/" "value."), .name_strobj = &_Py_ID(__truediv__) },
    {"__rtruediv__", offsetof(PyHeapTypeObject, as_number.nb_true_divide), (void *)(slot_nb_true_divide), wrap_binaryfunc_r, PyDoc_STR("__rtruediv__" "($self, value, /)\n--\n\nReturn value" "/" "self."), .name_strobj = &_Py_ID(__rtruediv__) },
    {"__ifloordiv__", offsetof(PyHeapTypeObject, as_number.nb_inplace_floor_divide), (void *)(slot_nb_inplace_floor_divide), wrap_binaryfunc, PyDoc_STR("__ifloordiv__" "($self, value, /)\n--\n\nReturn self" "//=" "value."), .name_strobj = &_Py_ID(__ifloordiv__) }
                                                                ,
    {"__itruediv__", offsetof(PyHeapTypeObject, as_number.nb_inplace_true_divide), (void *)(slot_nb_inplace_true_divide), wrap_binaryfunc, PyDoc_STR("__itruediv__" "($self, value, /)\n--\n\nReturn self" "/=" "value."), .name_strobj = &_Py_ID(__itruediv__) }
                                                              ,
    {"__index__", offsetof(PyHeapTypeObject, as_number.nb_index), (void *)(slot_nb_index), wrap_unaryfunc, PyDoc_STR("__index__($self, /)\n--\n\n" "Return self converted to an integer, if self is suitable " "for use as an index into a list."), .name_strobj = &_Py_ID(__index__) }


                                              ,
    {"__matmul__", offsetof(PyHeapTypeObject, as_number.nb_matrix_multiply), (void *)(slot_nb_matrix_multiply), wrap_binaryfunc_l, PyDoc_STR("__matmul__" "($self, value, /)\n--\n\nReturn self" "@" "value."), .name_strobj = &_Py_ID(__matmul__) }
                ,
    {"__rmatmul__", offsetof(PyHeapTypeObject, as_number.nb_matrix_multiply), (void *)(slot_nb_matrix_multiply), wrap_binaryfunc_r, PyDoc_STR("__rmatmul__" "($self, value, /)\n--\n\nReturn value" "@" "self."), .name_strobj = &_Py_ID(__rmatmul__) }
                 ,
    {"__imatmul__", offsetof(PyHeapTypeObject, as_number.nb_inplace_matrix_multiply), (void *)(slot_nb_inplace_matrix_multiply), wrap_binaryfunc, PyDoc_STR("__imatmul__" "($self, value, /)\n--\n\nReturn self" "@=" "value."), .name_strobj = &_Py_ID(__imatmul__) }
                                 ,
    {"__len__", offsetof(PyHeapTypeObject, as_mapping.mp_length), (void *)(slot_mp_length), wrap_lenfunc, PyDoc_STR("__len__($self, /)\n--\n\nReturn len(self)."), .name_strobj = &_Py_ID(__len__) }
                                                        ,
    {"__getitem__", offsetof(PyHeapTypeObject, as_mapping.mp_subscript), (void *)(slot_mp_subscript), wrap_binaryfunc, PyDoc_STR("__getitem__($self, key, /)\n--\n\nReturn self[key]."), .name_strobj = &_Py_ID(__getitem__) }

                                                                 ,
    {"__setitem__", offsetof(PyHeapTypeObject, as_mapping.mp_ass_subscript), (void *)(slot_mp_ass_subscript), wrap_objobjargproc, PyDoc_STR("__setitem__($self, key, value, /)\n--\n\nSet self[key] to value."), .name_strobj = &_Py_ID(__setitem__) }

                                                                              ,
    {"__delitem__", offsetof(PyHeapTypeObject, as_mapping.mp_ass_subscript), (void *)(slot_mp_ass_subscript), wrap_delitem, PyDoc_STR("__delitem__($self, key, /)\n--\n\nDelete self[key]."), .name_strobj = &_Py_ID(__delitem__) }

                                                                 ,

    {"__len__", offsetof(PyHeapTypeObject, as_sequence.sq_length), (void *)(slot_sq_length), wrap_lenfunc, PyDoc_STR("__len__($self, /)\n--\n\nReturn len(self)."), .name_strobj = &_Py_ID(__len__) }
                                                        ,





    {"__add__", offsetof(PyHeapTypeObject, as_sequence.sq_concat), (void *)(NULL), wrap_binaryfunc, PyDoc_STR("__add__($self, value, /)\n--\n\nReturn self+value."), .name_strobj = &_Py_ID(__add__) }
                                                                ,
    {"__mul__", offsetof(PyHeapTypeObject, as_sequence.sq_repeat), (void *)(NULL), wrap_indexargfunc, PyDoc_STR("__mul__($self, value, /)\n--\n\nReturn self*value."), .name_strobj = &_Py_ID(__mul__) }
                                                                ,
    {"__rmul__", offsetof(PyHeapTypeObject, as_sequence.sq_repeat), (void *)(NULL), wrap_indexargfunc, PyDoc_STR("__rmul__($self, value, /)\n--\n\nReturn value*self."), .name_strobj = &_Py_ID(__rmul__) }
                                                                 ,
    {"__getitem__", offsetof(PyHeapTypeObject, as_sequence.sq_item), (void *)(slot_sq_item), wrap_sq_item, PyDoc_STR("__getitem__($self, key, /)\n--\n\nReturn self[key]."), .name_strobj = &_Py_ID(__getitem__) }
                                                                 ,
    {"__setitem__", offsetof(PyHeapTypeObject, as_sequence.sq_ass_item), (void *)(slot_sq_ass_item), wrap_sq_setitem, PyDoc_STR("__setitem__($self, key, value, /)\n--\n\nSet self[key] to value."), .name_strobj = &_Py_ID(__setitem__) }
                                                                              ,
    {"__delitem__", offsetof(PyHeapTypeObject, as_sequence.sq_ass_item), (void *)(slot_sq_ass_item), wrap_sq_delitem, PyDoc_STR("__delitem__($self, key, /)\n--\n\nDelete self[key]."), .name_strobj = &_Py_ID(__delitem__) }
                                                                 ,
    {"__contains__", offsetof(PyHeapTypeObject, as_sequence.sq_contains), (void *)(slot_sq_contains), wrap_objobjproc, PyDoc_STR("__contains__($self, key, /)\n--\n\nReturn bool(key in self)."), .name_strobj = &_Py_ID(__contains__) }
                                                                          ,
    {"__iadd__", offsetof(PyHeapTypeObject, as_sequence.sq_inplace_concat), (void *)(NULL), wrap_binaryfunc, PyDoc_STR("__iadd__($self, value, /)\n--\n\nImplement self+=value."), .name_strobj = &_Py_ID(__iadd__) }

                                                                     ,
    {"__imul__", offsetof(PyHeapTypeObject, as_sequence.sq_inplace_repeat), (void *)(NULL), wrap_indexargfunc, PyDoc_STR("__imul__($self, value, /)\n--\n\nImplement self*=value."), .name_strobj = &_Py_ID(__imul__) }

                                                                     ,

