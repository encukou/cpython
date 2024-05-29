// This file is generated by Tools/cases_generator/uop_metadata_generator.py
// from:
//   Python/bytecodes.c
// Do not edit!

#ifndef Py_CORE_UOP_METADATA_H
#define Py_CORE_UOP_METADATA_H
#ifdef __cplusplus
extern "C" {
#endif

#include <stdint.h>
#include "pycore_uop_ids.h"
extern const uint16_t _PyUop_Flags[MAX_UOP_ID+1];
extern const uint8_t _PyUop_Replication[MAX_UOP_ID+1];
extern const char * const _PyOpcode_uop_name[MAX_UOP_ID+1];

extern int _PyUop_num_popped(int opcode, int oparg);

#ifdef NEED_OPCODE_METADATA
const uint16_t _PyUop_Flags[MAX_UOP_ID+1] = {
    [_NOP] = HAS_PURE_FLAG,
    [_RESUME_CHECK] = HAS_DEOPT_FLAG,
    [_LOAD_FAST_CHECK] = HAS_ARG_FLAG | HAS_LOCAL_FLAG | HAS_ERROR_FLAG | HAS_ESCAPES_FLAG,
    [_LOAD_FAST_0] = HAS_LOCAL_FLAG | HAS_PURE_FLAG,
    [_LOAD_FAST_1] = HAS_LOCAL_FLAG | HAS_PURE_FLAG,
    [_LOAD_FAST_2] = HAS_LOCAL_FLAG | HAS_PURE_FLAG,
    [_LOAD_FAST_3] = HAS_LOCAL_FLAG | HAS_PURE_FLAG,
    [_LOAD_FAST_4] = HAS_LOCAL_FLAG | HAS_PURE_FLAG,
    [_LOAD_FAST_5] = HAS_LOCAL_FLAG | HAS_PURE_FLAG,
    [_LOAD_FAST_6] = HAS_LOCAL_FLAG | HAS_PURE_FLAG,
    [_LOAD_FAST_7] = HAS_LOCAL_FLAG | HAS_PURE_FLAG,
    [_LOAD_FAST] = HAS_ARG_FLAG | HAS_LOCAL_FLAG | HAS_PURE_FLAG,
    [_LOAD_FAST_AND_CLEAR] = HAS_ARG_FLAG | HAS_LOCAL_FLAG,
    [_LOAD_FAST_LOAD_FAST] = HAS_ARG_FLAG | HAS_LOCAL_FLAG,
    [_LOAD_CONST] = HAS_ARG_FLAG | HAS_CONST_FLAG | HAS_PURE_FLAG,
    [_STORE_FAST_0] = HAS_LOCAL_FLAG,
    [_STORE_FAST_1] = HAS_LOCAL_FLAG,
    [_STORE_FAST_2] = HAS_LOCAL_FLAG,
    [_STORE_FAST_3] = HAS_LOCAL_FLAG,
    [_STORE_FAST_4] = HAS_LOCAL_FLAG,
    [_STORE_FAST_5] = HAS_LOCAL_FLAG,
    [_STORE_FAST_6] = HAS_LOCAL_FLAG,
    [_STORE_FAST_7] = HAS_LOCAL_FLAG,
    [_STORE_FAST] = HAS_ARG_FLAG | HAS_LOCAL_FLAG,
    [_STORE_FAST_LOAD_FAST] = HAS_ARG_FLAG | HAS_LOCAL_FLAG,
    [_STORE_FAST_STORE_FAST] = HAS_ARG_FLAG | HAS_LOCAL_FLAG,
    [_POP_TOP] = HAS_PURE_FLAG,
    [_PUSH_NULL] = HAS_PURE_FLAG,
    [_END_SEND] = HAS_PURE_FLAG,
    [_UNARY_NEGATIVE] = HAS_ERROR_FLAG | HAS_ESCAPES_FLAG,
    [_UNARY_NOT] = HAS_PURE_FLAG,
    [_TO_BOOL] = HAS_ERROR_FLAG | HAS_ESCAPES_FLAG,
    [_TO_BOOL_BOOL] = HAS_EXIT_FLAG,
    [_TO_BOOL_INT] = HAS_EXIT_FLAG,
    [_TO_BOOL_LIST] = HAS_EXIT_FLAG,
    [_TO_BOOL_NONE] = HAS_EXIT_FLAG,
    [_TO_BOOL_STR] = HAS_EXIT_FLAG,
    [_REPLACE_WITH_TRUE] = 0,
    [_UNARY_INVERT] = HAS_ERROR_FLAG | HAS_ESCAPES_FLAG,
    [_GUARD_BOTH_INT] = HAS_EXIT_FLAG,
    [_GUARD_NOS_INT] = HAS_EXIT_FLAG,
    [_GUARD_TOS_INT] = HAS_EXIT_FLAG,
    [_BINARY_OP_MULTIPLY_INT] = HAS_ERROR_FLAG | HAS_PURE_FLAG,
    [_BINARY_OP_ADD_INT] = HAS_ERROR_FLAG | HAS_PURE_FLAG,
    [_BINARY_OP_SUBTRACT_INT] = HAS_ERROR_FLAG | HAS_PURE_FLAG,
    [_GUARD_BOTH_FLOAT] = HAS_EXIT_FLAG,
    [_GUARD_NOS_FLOAT] = HAS_EXIT_FLAG,
    [_GUARD_TOS_FLOAT] = HAS_EXIT_FLAG,
    [_BINARY_OP_MULTIPLY_FLOAT] = HAS_PURE_FLAG,
    [_BINARY_OP_ADD_FLOAT] = HAS_PURE_FLAG,
    [_BINARY_OP_SUBTRACT_FLOAT] = HAS_PURE_FLAG,
    [_GUARD_BOTH_UNICODE] = HAS_EXIT_FLAG,
    [_BINARY_OP_ADD_UNICODE] = HAS_ERROR_FLAG | HAS_PURE_FLAG,
    [_BINARY_SUBSCR] = HAS_ERROR_FLAG | HAS_ESCAPES_FLAG,
    [_BINARY_SLICE] = HAS_ERROR_FLAG | HAS_ESCAPES_FLAG,
    [_STORE_SLICE] = HAS_ERROR_FLAG | HAS_ESCAPES_FLAG,
    [_BINARY_SUBSCR_LIST_INT] = HAS_DEOPT_FLAG,
    [_BINARY_SUBSCR_STR_INT] = HAS_DEOPT_FLAG,
    [_BINARY_SUBSCR_TUPLE_INT] = HAS_DEOPT_FLAG,
    [_BINARY_SUBSCR_DICT] = HAS_DEOPT_FLAG | HAS_ERROR_FLAG | HAS_ESCAPES_FLAG,
    [_LIST_APPEND] = HAS_ARG_FLAG | HAS_ERROR_FLAG,
    [_SET_ADD] = HAS_ARG_FLAG | HAS_ERROR_FLAG | HAS_ESCAPES_FLAG,
    [_STORE_SUBSCR] = HAS_ERROR_FLAG | HAS_ESCAPES_FLAG,
    [_STORE_SUBSCR_LIST_INT] = HAS_DEOPT_FLAG,
    [_STORE_SUBSCR_DICT] = HAS_DEOPT_FLAG | HAS_ERROR_FLAG | HAS_ESCAPES_FLAG,
    [_DELETE_SUBSCR] = HAS_ERROR_FLAG | HAS_ESCAPES_FLAG,
    [_CALL_INTRINSIC_1] = HAS_ARG_FLAG | HAS_ERROR_FLAG | HAS_ESCAPES_FLAG,
    [_CALL_INTRINSIC_2] = HAS_ARG_FLAG | HAS_ERROR_FLAG | HAS_ESCAPES_FLAG,
    [_POP_FRAME] = 0,
    [_GET_AITER] = HAS_ERROR_FLAG | HAS_ESCAPES_FLAG,
    [_GET_ANEXT] = HAS_ERROR_FLAG | HAS_ERROR_NO_POP_FLAG | HAS_ESCAPES_FLAG,
    [_GET_AWAITABLE] = HAS_ARG_FLAG | HAS_ERROR_FLAG | HAS_ESCAPES_FLAG,
    [_YIELD_VALUE] = HAS_ARG_FLAG | HAS_ESCAPES_FLAG,
    [_POP_EXCEPT] = HAS_ESCAPES_FLAG,
    [_LOAD_COMMON_CONSTANT] = HAS_ARG_FLAG,
    [_LOAD_BUILD_CLASS] = HAS_ERROR_FLAG | HAS_ESCAPES_FLAG,
    [_STORE_NAME] = HAS_ARG_FLAG | HAS_NAME_FLAG | HAS_ERROR_FLAG | HAS_ESCAPES_FLAG,
    [_DELETE_NAME] = HAS_ARG_FLAG | HAS_NAME_FLAG | HAS_ERROR_FLAG | HAS_ERROR_NO_POP_FLAG | HAS_ESCAPES_FLAG,
    [_UNPACK_SEQUENCE] = HAS_ARG_FLAG | HAS_ERROR_FLAG | HAS_ESCAPES_FLAG,
    [_UNPACK_SEQUENCE_TWO_TUPLE] = HAS_ARG_FLAG | HAS_DEOPT_FLAG,
    [_UNPACK_SEQUENCE_TUPLE] = HAS_ARG_FLAG | HAS_DEOPT_FLAG,
    [_UNPACK_SEQUENCE_LIST] = HAS_ARG_FLAG | HAS_DEOPT_FLAG,
    [_UNPACK_EX] = HAS_ARG_FLAG | HAS_ERROR_FLAG | HAS_ESCAPES_FLAG,
    [_STORE_ATTR] = HAS_ARG_FLAG | HAS_NAME_FLAG | HAS_ERROR_FLAG | HAS_ESCAPES_FLAG,
    [_DELETE_ATTR] = HAS_ARG_FLAG | HAS_NAME_FLAG | HAS_ERROR_FLAG | HAS_ESCAPES_FLAG,
    [_STORE_GLOBAL] = HAS_ARG_FLAG | HAS_NAME_FLAG | HAS_ERROR_FLAG | HAS_ESCAPES_FLAG,
    [_DELETE_GLOBAL] = HAS_ARG_FLAG | HAS_NAME_FLAG | HAS_ERROR_FLAG | HAS_ERROR_NO_POP_FLAG | HAS_ESCAPES_FLAG,
    [_LOAD_LOCALS] = HAS_ERROR_FLAG | HAS_ESCAPES_FLAG,
    [_LOAD_FROM_DICT_OR_GLOBALS] = HAS_ARG_FLAG | HAS_NAME_FLAG | HAS_ERROR_FLAG | HAS_ERROR_NO_POP_FLAG | HAS_ESCAPES_FLAG,
    [_LOAD_GLOBAL] = HAS_ARG_FLAG | HAS_NAME_FLAG | HAS_ERROR_FLAG | HAS_ESCAPES_FLAG,
    [_GUARD_GLOBALS_VERSION] = HAS_DEOPT_FLAG,
    [_GUARD_BUILTINS_VERSION] = HAS_DEOPT_FLAG,
    [_LOAD_GLOBAL_MODULE] = HAS_ARG_FLAG | HAS_DEOPT_FLAG,
    [_LOAD_GLOBAL_BUILTINS] = HAS_ARG_FLAG | HAS_DEOPT_FLAG,
    [_DELETE_FAST] = HAS_ARG_FLAG | HAS_LOCAL_FLAG | HAS_ERROR_FLAG | HAS_ESCAPES_FLAG,
    [_MAKE_CELL] = HAS_ARG_FLAG | HAS_FREE_FLAG | HAS_ERROR_FLAG | HAS_ERROR_NO_POP_FLAG,
    [_DELETE_DEREF] = HAS_ARG_FLAG | HAS_FREE_FLAG | HAS_ERROR_FLAG | HAS_ERROR_NO_POP_FLAG | HAS_ESCAPES_FLAG,
    [_LOAD_FROM_DICT_OR_DEREF] = HAS_ARG_FLAG | HAS_FREE_FLAG | HAS_ERROR_FLAG | HAS_ERROR_NO_POP_FLAG | HAS_ESCAPES_FLAG,
    [_LOAD_DEREF] = HAS_ARG_FLAG | HAS_FREE_FLAG | HAS_ERROR_FLAG | HAS_ESCAPES_FLAG,
    [_STORE_DEREF] = HAS_ARG_FLAG | HAS_FREE_FLAG | HAS_ESCAPES_FLAG,
    [_COPY_FREE_VARS] = HAS_ARG_FLAG,
    [_BUILD_STRING] = HAS_ARG_FLAG | HAS_ERROR_FLAG,
    [_BUILD_TUPLE] = HAS_ARG_FLAG | HAS_ERROR_FLAG,
    [_BUILD_LIST] = HAS_ARG_FLAG | HAS_ERROR_FLAG,
    [_LIST_EXTEND] = HAS_ARG_FLAG | HAS_ERROR_FLAG | HAS_ESCAPES_FLAG,
    [_SET_UPDATE] = HAS_ARG_FLAG | HAS_ERROR_FLAG | HAS_ESCAPES_FLAG,
    [_BUILD_MAP] = HAS_ARG_FLAG | HAS_ERROR_FLAG | HAS_ESCAPES_FLAG,
    [_SETUP_ANNOTATIONS] = HAS_ERROR_FLAG | HAS_ESCAPES_FLAG,
    [_BUILD_CONST_KEY_MAP] = HAS_ARG_FLAG | HAS_ERROR_FLAG | HAS_ESCAPES_FLAG,
    [_DICT_UPDATE] = HAS_ARG_FLAG | HAS_ERROR_FLAG | HAS_ESCAPES_FLAG,
    [_DICT_MERGE] = HAS_ARG_FLAG | HAS_ERROR_FLAG | HAS_ESCAPES_FLAG,
    [_MAP_ADD] = HAS_ARG_FLAG | HAS_ERROR_FLAG | HAS_ESCAPES_FLAG,
    [_LOAD_SUPER_ATTR_ATTR] = HAS_ARG_FLAG | HAS_NAME_FLAG | HAS_DEOPT_FLAG | HAS_ERROR_FLAG | HAS_ESCAPES_FLAG,
    [_LOAD_SUPER_ATTR_METHOD] = HAS_ARG_FLAG | HAS_NAME_FLAG | HAS_DEOPT_FLAG | HAS_ERROR_FLAG | HAS_ESCAPES_FLAG,
    [_LOAD_ATTR] = HAS_ARG_FLAG | HAS_NAME_FLAG | HAS_ERROR_FLAG | HAS_ESCAPES_FLAG,
    [_GUARD_TYPE_VERSION] = HAS_EXIT_FLAG,
    [_CHECK_MANAGED_OBJECT_HAS_VALUES] = HAS_DEOPT_FLAG,
    [_LOAD_ATTR_INSTANCE_VALUE_0] = HAS_DEOPT_FLAG,
    [_LOAD_ATTR_INSTANCE_VALUE_1] = HAS_DEOPT_FLAG,
    [_LOAD_ATTR_INSTANCE_VALUE] = HAS_ARG_FLAG | HAS_DEOPT_FLAG | HAS_OPARG_AND_1_FLAG,
    [_CHECK_ATTR_MODULE] = HAS_DEOPT_FLAG,
    [_LOAD_ATTR_MODULE] = HAS_ARG_FLAG | HAS_DEOPT_FLAG,
    [_CHECK_ATTR_WITH_HINT] = HAS_DEOPT_FLAG,
    [_LOAD_ATTR_WITH_HINT] = HAS_ARG_FLAG | HAS_NAME_FLAG | HAS_DEOPT_FLAG,
    [_LOAD_ATTR_SLOT_0] = HAS_DEOPT_FLAG,
    [_LOAD_ATTR_SLOT_1] = HAS_DEOPT_FLAG,
    [_LOAD_ATTR_SLOT] = HAS_ARG_FLAG | HAS_DEOPT_FLAG | HAS_OPARG_AND_1_FLAG,
    [_CHECK_ATTR_CLASS] = HAS_DEOPT_FLAG,
    [_LOAD_ATTR_CLASS_0] = 0,
    [_LOAD_ATTR_CLASS_1] = 0,
    [_LOAD_ATTR_CLASS] = HAS_ARG_FLAG | HAS_OPARG_AND_1_FLAG,
    [_GUARD_DORV_NO_DICT] = HAS_DEOPT_FLAG,
    [_STORE_ATTR_INSTANCE_VALUE] = 0,
    [_STORE_ATTR_WITH_HINT] = HAS_ARG_FLAG | HAS_NAME_FLAG | HAS_DEOPT_FLAG | HAS_ESCAPES_FLAG,
    [_STORE_ATTR_SLOT] = 0,
    [_COMPARE_OP] = HAS_ARG_FLAG | HAS_ERROR_FLAG | HAS_ESCAPES_FLAG,
    [_COMPARE_OP_FLOAT] = HAS_ARG_FLAG,
    [_COMPARE_OP_INT] = HAS_ARG_FLAG | HAS_DEOPT_FLAG,
    [_COMPARE_OP_STR] = HAS_ARG_FLAG,
    [_IS_OP] = HAS_ARG_FLAG,
    [_CONTAINS_OP] = HAS_ARG_FLAG | HAS_ERROR_FLAG | HAS_ESCAPES_FLAG,
    [_CONTAINS_OP_SET] = HAS_ARG_FLAG | HAS_DEOPT_FLAG | HAS_ERROR_FLAG | HAS_ESCAPES_FLAG,
    [_CONTAINS_OP_DICT] = HAS_ARG_FLAG | HAS_DEOPT_FLAG | HAS_ERROR_FLAG | HAS_ESCAPES_FLAG,
    [_CHECK_EG_MATCH] = HAS_ERROR_FLAG | HAS_ESCAPES_FLAG,
    [_CHECK_EXC_MATCH] = HAS_ERROR_FLAG | HAS_ESCAPES_FLAG,
    [_IS_NONE] = 0,
    [_GET_LEN] = HAS_ERROR_FLAG | HAS_ESCAPES_FLAG,
    [_MATCH_CLASS] = HAS_ARG_FLAG | HAS_ERROR_FLAG | HAS_ESCAPES_FLAG,
    [_MATCH_MAPPING] = 0,
    [_MATCH_SEQUENCE] = 0,
    [_MATCH_KEYS] = HAS_ERROR_FLAG | HAS_ESCAPES_FLAG,
    [_GET_ITER] = HAS_ERROR_FLAG | HAS_ESCAPES_FLAG,
    [_GET_YIELD_FROM_ITER] = HAS_ERROR_FLAG | HAS_ERROR_NO_POP_FLAG | HAS_ESCAPES_FLAG,
    [_FOR_ITER_TIER_TWO] = HAS_DEOPT_FLAG | HAS_ERROR_FLAG | HAS_ERROR_NO_POP_FLAG | HAS_ESCAPES_FLAG,
    [_ITER_CHECK_LIST] = HAS_EXIT_FLAG,
    [_GUARD_NOT_EXHAUSTED_LIST] = HAS_EXIT_FLAG,
    [_ITER_NEXT_LIST] = 0,
    [_ITER_CHECK_TUPLE] = HAS_EXIT_FLAG,
    [_GUARD_NOT_EXHAUSTED_TUPLE] = HAS_EXIT_FLAG,
    [_ITER_NEXT_TUPLE] = 0,
    [_ITER_CHECK_RANGE] = HAS_EXIT_FLAG,
    [_GUARD_NOT_EXHAUSTED_RANGE] = HAS_EXIT_FLAG,
    [_ITER_NEXT_RANGE] = HAS_ERROR_FLAG,
    [_FOR_ITER_GEN_FRAME] = HAS_ARG_FLAG | HAS_DEOPT_FLAG,
    [_WITH_EXCEPT_START] = HAS_ERROR_FLAG | HAS_ESCAPES_FLAG,
    [_PUSH_EXC_INFO] = 0,
    [_GUARD_DORV_VALUES_INST_ATTR_FROM_DICT] = HAS_DEOPT_FLAG,
    [_GUARD_KEYS_VERSION] = HAS_DEOPT_FLAG,
    [_LOAD_ATTR_METHOD_WITH_VALUES] = HAS_ARG_FLAG,
    [_LOAD_ATTR_METHOD_NO_DICT] = HAS_ARG_FLAG,
    [_LOAD_ATTR_NONDESCRIPTOR_WITH_VALUES] = HAS_ARG_FLAG,
    [_LOAD_ATTR_NONDESCRIPTOR_NO_DICT] = HAS_ARG_FLAG,
    [_CHECK_ATTR_METHOD_LAZY_DICT] = HAS_DEOPT_FLAG,
    [_LOAD_ATTR_METHOD_LAZY_DICT] = HAS_ARG_FLAG,
    [_CHECK_PERIODIC] = HAS_EVAL_BREAK_FLAG,
    [_PY_FRAME_GENERAL] = HAS_ARG_FLAG | HAS_ERROR_FLAG | HAS_ERROR_NO_POP_FLAG | HAS_ESCAPES_FLAG,
    [_CHECK_FUNCTION_VERSION] = HAS_ARG_FLAG | HAS_EXIT_FLAG,
    [_CHECK_METHOD_VERSION] = HAS_ARG_FLAG | HAS_EXIT_FLAG,
    [_EXPAND_METHOD] = HAS_ARG_FLAG,
    [_CHECK_IS_NOT_PY_CALLABLE] = HAS_ARG_FLAG | HAS_EXIT_FLAG,
    [_CALL_NON_PY_GENERAL] = HAS_ARG_FLAG | HAS_ERROR_FLAG | HAS_ESCAPES_FLAG,
    [_CHECK_CALL_BOUND_METHOD_EXACT_ARGS] = HAS_ARG_FLAG | HAS_EXIT_FLAG,
    [_INIT_CALL_BOUND_METHOD_EXACT_ARGS] = HAS_ARG_FLAG,
    [_CHECK_PEP_523] = HAS_DEOPT_FLAG,
    [_CHECK_FUNCTION_EXACT_ARGS] = HAS_ARG_FLAG | HAS_EXIT_FLAG,
    [_CHECK_STACK_SPACE] = HAS_ARG_FLAG | HAS_DEOPT_FLAG,
    [_INIT_CALL_PY_EXACT_ARGS_0] = HAS_PURE_FLAG,
    [_INIT_CALL_PY_EXACT_ARGS_1] = HAS_PURE_FLAG,
    [_INIT_CALL_PY_EXACT_ARGS_2] = HAS_PURE_FLAG,
    [_INIT_CALL_PY_EXACT_ARGS_3] = HAS_PURE_FLAG,
    [_INIT_CALL_PY_EXACT_ARGS_4] = HAS_PURE_FLAG,
    [_INIT_CALL_PY_EXACT_ARGS] = HAS_ARG_FLAG | HAS_PURE_FLAG,
    [_PUSH_FRAME] = 0,
    [_CALL_TYPE_1] = HAS_ARG_FLAG | HAS_DEOPT_FLAG,
    [_CALL_STR_1] = HAS_ARG_FLAG | HAS_DEOPT_FLAG | HAS_ERROR_FLAG | HAS_ESCAPES_FLAG,
    [_CALL_TUPLE_1] = HAS_ARG_FLAG | HAS_DEOPT_FLAG | HAS_ERROR_FLAG | HAS_ESCAPES_FLAG,
    [_EXIT_INIT_CHECK] = HAS_ERROR_FLAG | HAS_ERROR_NO_POP_FLAG | HAS_ESCAPES_FLAG,
    [_CALL_BUILTIN_CLASS] = HAS_ARG_FLAG | HAS_DEOPT_FLAG | HAS_ERROR_FLAG | HAS_ESCAPES_FLAG,
    [_CALL_BUILTIN_O] = HAS_ARG_FLAG | HAS_DEOPT_FLAG | HAS_ERROR_FLAG | HAS_ESCAPES_FLAG,
    [_CALL_BUILTIN_FAST] = HAS_ARG_FLAG | HAS_DEOPT_FLAG | HAS_ERROR_FLAG | HAS_ESCAPES_FLAG,
    [_CALL_BUILTIN_FAST_WITH_KEYWORDS] = HAS_ARG_FLAG | HAS_DEOPT_FLAG | HAS_ERROR_FLAG | HAS_ESCAPES_FLAG,
    [_CALL_LEN] = HAS_ARG_FLAG | HAS_DEOPT_FLAG | HAS_ERROR_FLAG | HAS_ERROR_NO_POP_FLAG | HAS_ESCAPES_FLAG,
    [_CALL_ISINSTANCE] = HAS_ARG_FLAG | HAS_DEOPT_FLAG | HAS_ERROR_FLAG | HAS_ERROR_NO_POP_FLAG | HAS_ESCAPES_FLAG,
    [_CALL_METHOD_DESCRIPTOR_O] = HAS_ARG_FLAG | HAS_DEOPT_FLAG | HAS_ERROR_FLAG | HAS_ESCAPES_FLAG,
    [_CALL_METHOD_DESCRIPTOR_FAST_WITH_KEYWORDS] = HAS_ARG_FLAG | HAS_DEOPT_FLAG | HAS_ERROR_FLAG | HAS_ESCAPES_FLAG,
    [_CALL_METHOD_DESCRIPTOR_NOARGS] = HAS_ARG_FLAG | HAS_DEOPT_FLAG | HAS_ERROR_FLAG | HAS_ESCAPES_FLAG,
    [_CALL_METHOD_DESCRIPTOR_FAST] = HAS_ARG_FLAG | HAS_DEOPT_FLAG | HAS_ERROR_FLAG | HAS_ESCAPES_FLAG,
    [_MAKE_FUNCTION] = HAS_ERROR_FLAG | HAS_ERROR_NO_POP_FLAG | HAS_ESCAPES_FLAG,
    [_SET_FUNCTION_ATTRIBUTE] = HAS_ARG_FLAG | HAS_ESCAPES_FLAG,
    [_RETURN_GENERATOR] = HAS_ERROR_FLAG | HAS_ERROR_NO_POP_FLAG | HAS_ESCAPES_FLAG,
    [_BUILD_SLICE] = HAS_ARG_FLAG | HAS_ERROR_FLAG,
    [_CONVERT_VALUE] = HAS_ARG_FLAG | HAS_ERROR_FLAG,
    [_FORMAT_SIMPLE] = HAS_ERROR_FLAG | HAS_ESCAPES_FLAG,
    [_FORMAT_WITH_SPEC] = HAS_ERROR_FLAG | HAS_ESCAPES_FLAG,
    [_COPY] = HAS_ARG_FLAG | HAS_PURE_FLAG,
    [_BINARY_OP] = HAS_ARG_FLAG | HAS_ERROR_FLAG | HAS_ESCAPES_FLAG,
    [_SWAP] = HAS_ARG_FLAG | HAS_PURE_FLAG,
    [_GUARD_IS_TRUE_POP] = HAS_EXIT_FLAG,
    [_GUARD_IS_FALSE_POP] = HAS_EXIT_FLAG,
    [_GUARD_IS_NONE_POP] = HAS_EXIT_FLAG,
    [_GUARD_IS_NOT_NONE_POP] = HAS_EXIT_FLAG,
    [_JUMP_TO_TOP] = 0,
    [_SET_IP] = 0,
    [_CHECK_STACK_SPACE_OPERAND] = HAS_DEOPT_FLAG,
    [_SAVE_RETURN_OFFSET] = HAS_ARG_FLAG,
    [_EXIT_TRACE] = 0,
    [_CHECK_VALIDITY] = HAS_DEOPT_FLAG,
    [_LOAD_CONST_INLINE] = HAS_PURE_FLAG,
    [_LOAD_CONST_INLINE_BORROW] = HAS_PURE_FLAG,
    [_POP_TOP_LOAD_CONST_INLINE_BORROW] = HAS_PURE_FLAG,
    [_LOAD_CONST_INLINE_WITH_NULL] = HAS_PURE_FLAG,
    [_LOAD_CONST_INLINE_BORROW_WITH_NULL] = HAS_PURE_FLAG,
    [_CHECK_FUNCTION] = HAS_DEOPT_FLAG,
    [_INTERNAL_INCREMENT_OPT_COUNTER] = 0,
    [_COLD_EXIT] = HAS_ARG_FLAG | HAS_ESCAPES_FLAG,
    [_DYNAMIC_EXIT] = HAS_ARG_FLAG | HAS_ESCAPES_FLAG,
    [_START_EXECUTOR] = HAS_DEOPT_FLAG,
    [_FATAL_ERROR] = 0,
    [_CHECK_VALIDITY_AND_SET_IP] = HAS_DEOPT_FLAG,
    [_DEOPT] = 0,
    [_ERROR_POP_N] = HAS_ARG_FLAG,
    [_TIER2_RESUME_CHECK] = HAS_DEOPT_FLAG,
};

const uint8_t _PyUop_Replication[MAX_UOP_ID+1] = {
    [_LOAD_FAST] = 8,
    [_STORE_FAST] = 8,
    [_INIT_CALL_PY_EXACT_ARGS] = 5,
};

const char *const _PyOpcode_uop_name[MAX_UOP_ID+1] = {
    [_BINARY_OP] = "_BINARY_OP",
    [_BINARY_OP_ADD_FLOAT] = "_BINARY_OP_ADD_FLOAT",
    [_BINARY_OP_ADD_INT] = "_BINARY_OP_ADD_INT",
    [_BINARY_OP_ADD_UNICODE] = "_BINARY_OP_ADD_UNICODE",
    [_BINARY_OP_MULTIPLY_FLOAT] = "_BINARY_OP_MULTIPLY_FLOAT",
    [_BINARY_OP_MULTIPLY_INT] = "_BINARY_OP_MULTIPLY_INT",
    [_BINARY_OP_SUBTRACT_FLOAT] = "_BINARY_OP_SUBTRACT_FLOAT",
    [_BINARY_OP_SUBTRACT_INT] = "_BINARY_OP_SUBTRACT_INT",
    [_BINARY_SLICE] = "_BINARY_SLICE",
    [_BINARY_SUBSCR] = "_BINARY_SUBSCR",
    [_BINARY_SUBSCR_DICT] = "_BINARY_SUBSCR_DICT",
    [_BINARY_SUBSCR_LIST_INT] = "_BINARY_SUBSCR_LIST_INT",
    [_BINARY_SUBSCR_STR_INT] = "_BINARY_SUBSCR_STR_INT",
    [_BINARY_SUBSCR_TUPLE_INT] = "_BINARY_SUBSCR_TUPLE_INT",
    [_BUILD_CONST_KEY_MAP] = "_BUILD_CONST_KEY_MAP",
    [_BUILD_LIST] = "_BUILD_LIST",
    [_BUILD_MAP] = "_BUILD_MAP",
    [_BUILD_SLICE] = "_BUILD_SLICE",
    [_BUILD_STRING] = "_BUILD_STRING",
    [_BUILD_TUPLE] = "_BUILD_TUPLE",
    [_CALL_BUILTIN_CLASS] = "_CALL_BUILTIN_CLASS",
    [_CALL_BUILTIN_FAST] = "_CALL_BUILTIN_FAST",
    [_CALL_BUILTIN_FAST_WITH_KEYWORDS] = "_CALL_BUILTIN_FAST_WITH_KEYWORDS",
    [_CALL_BUILTIN_O] = "_CALL_BUILTIN_O",
    [_CALL_INTRINSIC_1] = "_CALL_INTRINSIC_1",
    [_CALL_INTRINSIC_2] = "_CALL_INTRINSIC_2",
    [_CALL_ISINSTANCE] = "_CALL_ISINSTANCE",
    [_CALL_LEN] = "_CALL_LEN",
    [_CALL_METHOD_DESCRIPTOR_FAST] = "_CALL_METHOD_DESCRIPTOR_FAST",
    [_CALL_METHOD_DESCRIPTOR_FAST_WITH_KEYWORDS] = "_CALL_METHOD_DESCRIPTOR_FAST_WITH_KEYWORDS",
    [_CALL_METHOD_DESCRIPTOR_NOARGS] = "_CALL_METHOD_DESCRIPTOR_NOARGS",
    [_CALL_METHOD_DESCRIPTOR_O] = "_CALL_METHOD_DESCRIPTOR_O",
    [_CALL_NON_PY_GENERAL] = "_CALL_NON_PY_GENERAL",
    [_CALL_STR_1] = "_CALL_STR_1",
    [_CALL_TUPLE_1] = "_CALL_TUPLE_1",
    [_CALL_TYPE_1] = "_CALL_TYPE_1",
    [_CHECK_ATTR_CLASS] = "_CHECK_ATTR_CLASS",
    [_CHECK_ATTR_METHOD_LAZY_DICT] = "_CHECK_ATTR_METHOD_LAZY_DICT",
    [_CHECK_ATTR_MODULE] = "_CHECK_ATTR_MODULE",
    [_CHECK_ATTR_WITH_HINT] = "_CHECK_ATTR_WITH_HINT",
    [_CHECK_CALL_BOUND_METHOD_EXACT_ARGS] = "_CHECK_CALL_BOUND_METHOD_EXACT_ARGS",
    [_CHECK_EG_MATCH] = "_CHECK_EG_MATCH",
    [_CHECK_EXC_MATCH] = "_CHECK_EXC_MATCH",
    [_CHECK_FUNCTION] = "_CHECK_FUNCTION",
    [_CHECK_FUNCTION_EXACT_ARGS] = "_CHECK_FUNCTION_EXACT_ARGS",
    [_CHECK_FUNCTION_VERSION] = "_CHECK_FUNCTION_VERSION",
    [_CHECK_IS_NOT_PY_CALLABLE] = "_CHECK_IS_NOT_PY_CALLABLE",
    [_CHECK_MANAGED_OBJECT_HAS_VALUES] = "_CHECK_MANAGED_OBJECT_HAS_VALUES",
    [_CHECK_METHOD_VERSION] = "_CHECK_METHOD_VERSION",
    [_CHECK_PEP_523] = "_CHECK_PEP_523",
    [_CHECK_PERIODIC] = "_CHECK_PERIODIC",
    [_CHECK_STACK_SPACE] = "_CHECK_STACK_SPACE",
    [_CHECK_STACK_SPACE_OPERAND] = "_CHECK_STACK_SPACE_OPERAND",
    [_CHECK_VALIDITY] = "_CHECK_VALIDITY",
    [_CHECK_VALIDITY_AND_SET_IP] = "_CHECK_VALIDITY_AND_SET_IP",
    [_COLD_EXIT] = "_COLD_EXIT",
    [_COMPARE_OP] = "_COMPARE_OP",
    [_COMPARE_OP_FLOAT] = "_COMPARE_OP_FLOAT",
    [_COMPARE_OP_INT] = "_COMPARE_OP_INT",
    [_COMPARE_OP_STR] = "_COMPARE_OP_STR",
    [_CONTAINS_OP] = "_CONTAINS_OP",
    [_CONTAINS_OP_DICT] = "_CONTAINS_OP_DICT",
    [_CONTAINS_OP_SET] = "_CONTAINS_OP_SET",
    [_CONVERT_VALUE] = "_CONVERT_VALUE",
    [_COPY] = "_COPY",
    [_COPY_FREE_VARS] = "_COPY_FREE_VARS",
    [_DELETE_ATTR] = "_DELETE_ATTR",
    [_DELETE_DEREF] = "_DELETE_DEREF",
    [_DELETE_FAST] = "_DELETE_FAST",
    [_DELETE_GLOBAL] = "_DELETE_GLOBAL",
    [_DELETE_NAME] = "_DELETE_NAME",
    [_DELETE_SUBSCR] = "_DELETE_SUBSCR",
    [_DEOPT] = "_DEOPT",
    [_DICT_MERGE] = "_DICT_MERGE",
    [_DICT_UPDATE] = "_DICT_UPDATE",
    [_DYNAMIC_EXIT] = "_DYNAMIC_EXIT",
    [_END_SEND] = "_END_SEND",
    [_ERROR_POP_N] = "_ERROR_POP_N",
    [_EXIT_INIT_CHECK] = "_EXIT_INIT_CHECK",
    [_EXIT_TRACE] = "_EXIT_TRACE",
    [_EXPAND_METHOD] = "_EXPAND_METHOD",
    [_FATAL_ERROR] = "_FATAL_ERROR",
    [_FORMAT_SIMPLE] = "_FORMAT_SIMPLE",
    [_FORMAT_WITH_SPEC] = "_FORMAT_WITH_SPEC",
    [_FOR_ITER_GEN_FRAME] = "_FOR_ITER_GEN_FRAME",
    [_FOR_ITER_TIER_TWO] = "_FOR_ITER_TIER_TWO",
    [_GET_AITER] = "_GET_AITER",
    [_GET_ANEXT] = "_GET_ANEXT",
    [_GET_AWAITABLE] = "_GET_AWAITABLE",
    [_GET_ITER] = "_GET_ITER",
    [_GET_LEN] = "_GET_LEN",
    [_GET_YIELD_FROM_ITER] = "_GET_YIELD_FROM_ITER",
    [_GUARD_BOTH_FLOAT] = "_GUARD_BOTH_FLOAT",
    [_GUARD_BOTH_INT] = "_GUARD_BOTH_INT",
    [_GUARD_BOTH_UNICODE] = "_GUARD_BOTH_UNICODE",
    [_GUARD_BUILTINS_VERSION] = "_GUARD_BUILTINS_VERSION",
    [_GUARD_DORV_NO_DICT] = "_GUARD_DORV_NO_DICT",
    [_GUARD_DORV_VALUES_INST_ATTR_FROM_DICT] = "_GUARD_DORV_VALUES_INST_ATTR_FROM_DICT",
    [_GUARD_GLOBALS_VERSION] = "_GUARD_GLOBALS_VERSION",
    [_GUARD_IS_FALSE_POP] = "_GUARD_IS_FALSE_POP",
    [_GUARD_IS_NONE_POP] = "_GUARD_IS_NONE_POP",
    [_GUARD_IS_NOT_NONE_POP] = "_GUARD_IS_NOT_NONE_POP",
    [_GUARD_IS_TRUE_POP] = "_GUARD_IS_TRUE_POP",
    [_GUARD_KEYS_VERSION] = "_GUARD_KEYS_VERSION",
    [_GUARD_NOS_FLOAT] = "_GUARD_NOS_FLOAT",
    [_GUARD_NOS_INT] = "_GUARD_NOS_INT",
    [_GUARD_NOT_EXHAUSTED_LIST] = "_GUARD_NOT_EXHAUSTED_LIST",
    [_GUARD_NOT_EXHAUSTED_RANGE] = "_GUARD_NOT_EXHAUSTED_RANGE",
    [_GUARD_NOT_EXHAUSTED_TUPLE] = "_GUARD_NOT_EXHAUSTED_TUPLE",
    [_GUARD_TOS_FLOAT] = "_GUARD_TOS_FLOAT",
    [_GUARD_TOS_INT] = "_GUARD_TOS_INT",
    [_GUARD_TYPE_VERSION] = "_GUARD_TYPE_VERSION",
    [_INIT_CALL_BOUND_METHOD_EXACT_ARGS] = "_INIT_CALL_BOUND_METHOD_EXACT_ARGS",
    [_INIT_CALL_PY_EXACT_ARGS] = "_INIT_CALL_PY_EXACT_ARGS",
    [_INIT_CALL_PY_EXACT_ARGS_0] = "_INIT_CALL_PY_EXACT_ARGS_0",
    [_INIT_CALL_PY_EXACT_ARGS_1] = "_INIT_CALL_PY_EXACT_ARGS_1",
    [_INIT_CALL_PY_EXACT_ARGS_2] = "_INIT_CALL_PY_EXACT_ARGS_2",
    [_INIT_CALL_PY_EXACT_ARGS_3] = "_INIT_CALL_PY_EXACT_ARGS_3",
    [_INIT_CALL_PY_EXACT_ARGS_4] = "_INIT_CALL_PY_EXACT_ARGS_4",
    [_INTERNAL_INCREMENT_OPT_COUNTER] = "_INTERNAL_INCREMENT_OPT_COUNTER",
    [_IS_NONE] = "_IS_NONE",
    [_IS_OP] = "_IS_OP",
    [_ITER_CHECK_LIST] = "_ITER_CHECK_LIST",
    [_ITER_CHECK_RANGE] = "_ITER_CHECK_RANGE",
    [_ITER_CHECK_TUPLE] = "_ITER_CHECK_TUPLE",
    [_ITER_NEXT_LIST] = "_ITER_NEXT_LIST",
    [_ITER_NEXT_RANGE] = "_ITER_NEXT_RANGE",
    [_ITER_NEXT_TUPLE] = "_ITER_NEXT_TUPLE",
    [_JUMP_TO_TOP] = "_JUMP_TO_TOP",
    [_LIST_APPEND] = "_LIST_APPEND",
    [_LIST_EXTEND] = "_LIST_EXTEND",
    [_LOAD_ATTR] = "_LOAD_ATTR",
    [_LOAD_ATTR_CLASS] = "_LOAD_ATTR_CLASS",
    [_LOAD_ATTR_CLASS_0] = "_LOAD_ATTR_CLASS_0",
    [_LOAD_ATTR_CLASS_1] = "_LOAD_ATTR_CLASS_1",
    [_LOAD_ATTR_INSTANCE_VALUE] = "_LOAD_ATTR_INSTANCE_VALUE",
    [_LOAD_ATTR_INSTANCE_VALUE_0] = "_LOAD_ATTR_INSTANCE_VALUE_0",
    [_LOAD_ATTR_INSTANCE_VALUE_1] = "_LOAD_ATTR_INSTANCE_VALUE_1",
    [_LOAD_ATTR_METHOD_LAZY_DICT] = "_LOAD_ATTR_METHOD_LAZY_DICT",
    [_LOAD_ATTR_METHOD_NO_DICT] = "_LOAD_ATTR_METHOD_NO_DICT",
    [_LOAD_ATTR_METHOD_WITH_VALUES] = "_LOAD_ATTR_METHOD_WITH_VALUES",
    [_LOAD_ATTR_MODULE] = "_LOAD_ATTR_MODULE",
    [_LOAD_ATTR_NONDESCRIPTOR_NO_DICT] = "_LOAD_ATTR_NONDESCRIPTOR_NO_DICT",
    [_LOAD_ATTR_NONDESCRIPTOR_WITH_VALUES] = "_LOAD_ATTR_NONDESCRIPTOR_WITH_VALUES",
    [_LOAD_ATTR_SLOT] = "_LOAD_ATTR_SLOT",
    [_LOAD_ATTR_SLOT_0] = "_LOAD_ATTR_SLOT_0",
    [_LOAD_ATTR_SLOT_1] = "_LOAD_ATTR_SLOT_1",
    [_LOAD_ATTR_WITH_HINT] = "_LOAD_ATTR_WITH_HINT",
    [_LOAD_BUILD_CLASS] = "_LOAD_BUILD_CLASS",
    [_LOAD_COMMON_CONSTANT] = "_LOAD_COMMON_CONSTANT",
    [_LOAD_CONST] = "_LOAD_CONST",
    [_LOAD_CONST_INLINE] = "_LOAD_CONST_INLINE",
    [_LOAD_CONST_INLINE_BORROW] = "_LOAD_CONST_INLINE_BORROW",
    [_LOAD_CONST_INLINE_BORROW_WITH_NULL] = "_LOAD_CONST_INLINE_BORROW_WITH_NULL",
    [_LOAD_CONST_INLINE_WITH_NULL] = "_LOAD_CONST_INLINE_WITH_NULL",
    [_LOAD_DEREF] = "_LOAD_DEREF",
    [_LOAD_FAST] = "_LOAD_FAST",
    [_LOAD_FAST_0] = "_LOAD_FAST_0",
    [_LOAD_FAST_1] = "_LOAD_FAST_1",
    [_LOAD_FAST_2] = "_LOAD_FAST_2",
    [_LOAD_FAST_3] = "_LOAD_FAST_3",
    [_LOAD_FAST_4] = "_LOAD_FAST_4",
    [_LOAD_FAST_5] = "_LOAD_FAST_5",
    [_LOAD_FAST_6] = "_LOAD_FAST_6",
    [_LOAD_FAST_7] = "_LOAD_FAST_7",
    [_LOAD_FAST_AND_CLEAR] = "_LOAD_FAST_AND_CLEAR",
    [_LOAD_FAST_CHECK] = "_LOAD_FAST_CHECK",
    [_LOAD_FAST_LOAD_FAST] = "_LOAD_FAST_LOAD_FAST",
    [_LOAD_FROM_DICT_OR_DEREF] = "_LOAD_FROM_DICT_OR_DEREF",
    [_LOAD_FROM_DICT_OR_GLOBALS] = "_LOAD_FROM_DICT_OR_GLOBALS",
    [_LOAD_GLOBAL] = "_LOAD_GLOBAL",
    [_LOAD_GLOBAL_BUILTINS] = "_LOAD_GLOBAL_BUILTINS",
    [_LOAD_GLOBAL_MODULE] = "_LOAD_GLOBAL_MODULE",
    [_LOAD_LOCALS] = "_LOAD_LOCALS",
    [_LOAD_SUPER_ATTR_ATTR] = "_LOAD_SUPER_ATTR_ATTR",
    [_LOAD_SUPER_ATTR_METHOD] = "_LOAD_SUPER_ATTR_METHOD",
    [_MAKE_CELL] = "_MAKE_CELL",
    [_MAKE_FUNCTION] = "_MAKE_FUNCTION",
    [_MAP_ADD] = "_MAP_ADD",
    [_MATCH_CLASS] = "_MATCH_CLASS",
    [_MATCH_KEYS] = "_MATCH_KEYS",
    [_MATCH_MAPPING] = "_MATCH_MAPPING",
    [_MATCH_SEQUENCE] = "_MATCH_SEQUENCE",
    [_NOP] = "_NOP",
    [_POP_EXCEPT] = "_POP_EXCEPT",
    [_POP_FRAME] = "_POP_FRAME",
    [_POP_TOP] = "_POP_TOP",
    [_POP_TOP_LOAD_CONST_INLINE_BORROW] = "_POP_TOP_LOAD_CONST_INLINE_BORROW",
    [_PUSH_EXC_INFO] = "_PUSH_EXC_INFO",
    [_PUSH_FRAME] = "_PUSH_FRAME",
    [_PUSH_NULL] = "_PUSH_NULL",
    [_PY_FRAME_GENERAL] = "_PY_FRAME_GENERAL",
    [_REPLACE_WITH_TRUE] = "_REPLACE_WITH_TRUE",
    [_RESUME_CHECK] = "_RESUME_CHECK",
    [_RETURN_GENERATOR] = "_RETURN_GENERATOR",
    [_SAVE_RETURN_OFFSET] = "_SAVE_RETURN_OFFSET",
    [_SETUP_ANNOTATIONS] = "_SETUP_ANNOTATIONS",
    [_SET_ADD] = "_SET_ADD",
    [_SET_FUNCTION_ATTRIBUTE] = "_SET_FUNCTION_ATTRIBUTE",
    [_SET_IP] = "_SET_IP",
    [_SET_UPDATE] = "_SET_UPDATE",
    [_START_EXECUTOR] = "_START_EXECUTOR",
    [_STORE_ATTR] = "_STORE_ATTR",
    [_STORE_ATTR_INSTANCE_VALUE] = "_STORE_ATTR_INSTANCE_VALUE",
    [_STORE_ATTR_SLOT] = "_STORE_ATTR_SLOT",
    [_STORE_ATTR_WITH_HINT] = "_STORE_ATTR_WITH_HINT",
    [_STORE_DEREF] = "_STORE_DEREF",
    [_STORE_FAST] = "_STORE_FAST",
    [_STORE_FAST_0] = "_STORE_FAST_0",
    [_STORE_FAST_1] = "_STORE_FAST_1",
    [_STORE_FAST_2] = "_STORE_FAST_2",
    [_STORE_FAST_3] = "_STORE_FAST_3",
    [_STORE_FAST_4] = "_STORE_FAST_4",
    [_STORE_FAST_5] = "_STORE_FAST_5",
    [_STORE_FAST_6] = "_STORE_FAST_6",
    [_STORE_FAST_7] = "_STORE_FAST_7",
    [_STORE_FAST_LOAD_FAST] = "_STORE_FAST_LOAD_FAST",
    [_STORE_FAST_STORE_FAST] = "_STORE_FAST_STORE_FAST",
    [_STORE_GLOBAL] = "_STORE_GLOBAL",
    [_STORE_NAME] = "_STORE_NAME",
    [_STORE_SLICE] = "_STORE_SLICE",
    [_STORE_SUBSCR] = "_STORE_SUBSCR",
    [_STORE_SUBSCR_DICT] = "_STORE_SUBSCR_DICT",
    [_STORE_SUBSCR_LIST_INT] = "_STORE_SUBSCR_LIST_INT",
    [_SWAP] = "_SWAP",
    [_TIER2_RESUME_CHECK] = "_TIER2_RESUME_CHECK",
    [_TO_BOOL] = "_TO_BOOL",
    [_TO_BOOL_BOOL] = "_TO_BOOL_BOOL",
    [_TO_BOOL_INT] = "_TO_BOOL_INT",
    [_TO_BOOL_LIST] = "_TO_BOOL_LIST",
    [_TO_BOOL_NONE] = "_TO_BOOL_NONE",
    [_TO_BOOL_STR] = "_TO_BOOL_STR",
    [_UNARY_INVERT] = "_UNARY_INVERT",
    [_UNARY_NEGATIVE] = "_UNARY_NEGATIVE",
    [_UNARY_NOT] = "_UNARY_NOT",
    [_UNPACK_EX] = "_UNPACK_EX",
    [_UNPACK_SEQUENCE] = "_UNPACK_SEQUENCE",
    [_UNPACK_SEQUENCE_LIST] = "_UNPACK_SEQUENCE_LIST",
    [_UNPACK_SEQUENCE_TUPLE] = "_UNPACK_SEQUENCE_TUPLE",
    [_UNPACK_SEQUENCE_TWO_TUPLE] = "_UNPACK_SEQUENCE_TWO_TUPLE",
    [_WITH_EXCEPT_START] = "_WITH_EXCEPT_START",
    [_YIELD_VALUE] = "_YIELD_VALUE",
};
int _PyUop_num_popped(int opcode, int oparg)
{
    switch(opcode) {
        case _NOP:
            return 0;
        case _RESUME_CHECK:
            return 0;
        case _LOAD_FAST_CHECK:
            return 0;
        case _LOAD_FAST_0:
            return 0;
        case _LOAD_FAST_1:
            return 0;
        case _LOAD_FAST_2:
            return 0;
        case _LOAD_FAST_3:
            return 0;
        case _LOAD_FAST_4:
            return 0;
        case _LOAD_FAST_5:
            return 0;
        case _LOAD_FAST_6:
            return 0;
        case _LOAD_FAST_7:
            return 0;
        case _LOAD_FAST:
            return 0;
        case _LOAD_FAST_AND_CLEAR:
            return 0;
        case _LOAD_FAST_LOAD_FAST:
            return 0;
        case _LOAD_CONST:
            return 0;
        case _STORE_FAST_0:
            return 1;
        case _STORE_FAST_1:
            return 1;
        case _STORE_FAST_2:
            return 1;
        case _STORE_FAST_3:
            return 1;
        case _STORE_FAST_4:
            return 1;
        case _STORE_FAST_5:
            return 1;
        case _STORE_FAST_6:
            return 1;
        case _STORE_FAST_7:
            return 1;
        case _STORE_FAST:
            return 1;
        case _STORE_FAST_LOAD_FAST:
            return 1;
        case _STORE_FAST_STORE_FAST:
            return 2;
        case _POP_TOP:
            return 1;
        case _PUSH_NULL:
            return 0;
        case _END_SEND:
            return 2;
        case _UNARY_NEGATIVE:
            return 1;
        case _UNARY_NOT:
            return 1;
        case _TO_BOOL:
            return 1;
        case _TO_BOOL_BOOL:
            return 1;
        case _TO_BOOL_INT:
            return 1;
        case _TO_BOOL_LIST:
            return 1;
        case _TO_BOOL_NONE:
            return 1;
        case _TO_BOOL_STR:
            return 1;
        case _REPLACE_WITH_TRUE:
            return 1;
        case _UNARY_INVERT:
            return 1;
        case _GUARD_BOTH_INT:
            return 2;
        case _GUARD_NOS_INT:
            return 2;
        case _GUARD_TOS_INT:
            return 1;
        case _BINARY_OP_MULTIPLY_INT:
            return 2;
        case _BINARY_OP_ADD_INT:
            return 2;
        case _BINARY_OP_SUBTRACT_INT:
            return 2;
        case _GUARD_BOTH_FLOAT:
            return 2;
        case _GUARD_NOS_FLOAT:
            return 2;
        case _GUARD_TOS_FLOAT:
            return 1;
        case _BINARY_OP_MULTIPLY_FLOAT:
            return 2;
        case _BINARY_OP_ADD_FLOAT:
            return 2;
        case _BINARY_OP_SUBTRACT_FLOAT:
            return 2;
        case _GUARD_BOTH_UNICODE:
            return 2;
        case _BINARY_OP_ADD_UNICODE:
            return 2;
        case _BINARY_SUBSCR:
            return 2;
        case _BINARY_SLICE:
            return 3;
        case _STORE_SLICE:
            return 4;
        case _BINARY_SUBSCR_LIST_INT:
            return 2;
        case _BINARY_SUBSCR_STR_INT:
            return 2;
        case _BINARY_SUBSCR_TUPLE_INT:
            return 2;
        case _BINARY_SUBSCR_DICT:
            return 2;
        case _LIST_APPEND:
            return 2 + (oparg-1);
        case _SET_ADD:
            return 2 + (oparg-1);
        case _STORE_SUBSCR:
            return 3;
        case _STORE_SUBSCR_LIST_INT:
            return 3;
        case _STORE_SUBSCR_DICT:
            return 3;
        case _DELETE_SUBSCR:
            return 2;
        case _CALL_INTRINSIC_1:
            return 1;
        case _CALL_INTRINSIC_2:
            return 2;
        case _POP_FRAME:
            return 1;
        case _GET_AITER:
            return 1;
        case _GET_ANEXT:
            return 1;
        case _GET_AWAITABLE:
            return 1;
        case _YIELD_VALUE:
            return 1;
        case _POP_EXCEPT:
            return 1;
        case _LOAD_COMMON_CONSTANT:
            return 0;
        case _LOAD_BUILD_CLASS:
            return 0;
        case _STORE_NAME:
            return 1;
        case _DELETE_NAME:
            return 0;
        case _UNPACK_SEQUENCE:
            return 1;
        case _UNPACK_SEQUENCE_TWO_TUPLE:
            return 1;
        case _UNPACK_SEQUENCE_TUPLE:
            return 1;
        case _UNPACK_SEQUENCE_LIST:
            return 1;
        case _UNPACK_EX:
            return 1;
        case _STORE_ATTR:
            return 2;
        case _DELETE_ATTR:
            return 1;
        case _STORE_GLOBAL:
            return 1;
        case _DELETE_GLOBAL:
            return 0;
        case _LOAD_LOCALS:
            return 0;
        case _LOAD_FROM_DICT_OR_GLOBALS:
            return 1;
        case _LOAD_GLOBAL:
            return 0;
        case _GUARD_GLOBALS_VERSION:
            return 0;
        case _GUARD_BUILTINS_VERSION:
            return 0;
        case _LOAD_GLOBAL_MODULE:
            return 0;
        case _LOAD_GLOBAL_BUILTINS:
            return 0;
        case _DELETE_FAST:
            return 0;
        case _MAKE_CELL:
            return 0;
        case _DELETE_DEREF:
            return 0;
        case _LOAD_FROM_DICT_OR_DEREF:
            return 1;
        case _LOAD_DEREF:
            return 0;
        case _STORE_DEREF:
            return 1;
        case _COPY_FREE_VARS:
            return 0;
        case _BUILD_STRING:
            return oparg;
        case _BUILD_TUPLE:
            return oparg;
        case _BUILD_LIST:
            return oparg;
        case _LIST_EXTEND:
            return 2 + (oparg-1);
        case _SET_UPDATE:
            return 2 + (oparg-1);
        case _BUILD_MAP:
            return oparg*2;
        case _SETUP_ANNOTATIONS:
            return 0;
        case _BUILD_CONST_KEY_MAP:
            return 1 + oparg;
        case _DICT_UPDATE:
            return 2 + (oparg - 1);
        case _DICT_MERGE:
            return 5 + (oparg - 1);
        case _MAP_ADD:
            return 3 + (oparg - 1);
        case _LOAD_SUPER_ATTR_ATTR:
            return 3;
        case _LOAD_SUPER_ATTR_METHOD:
            return 3;
        case _LOAD_ATTR:
            return 1;
        case _GUARD_TYPE_VERSION:
            return 1;
        case _CHECK_MANAGED_OBJECT_HAS_VALUES:
            return 1;
        case _LOAD_ATTR_INSTANCE_VALUE_0:
            return 1;
        case _LOAD_ATTR_INSTANCE_VALUE_1:
            return 1;
        case _LOAD_ATTR_INSTANCE_VALUE:
            return 1;
        case _CHECK_ATTR_MODULE:
            return 1;
        case _LOAD_ATTR_MODULE:
            return 1;
        case _CHECK_ATTR_WITH_HINT:
            return 1;
        case _LOAD_ATTR_WITH_HINT:
            return 1;
        case _LOAD_ATTR_SLOT_0:
            return 1;
        case _LOAD_ATTR_SLOT_1:
            return 1;
        case _LOAD_ATTR_SLOT:
            return 1;
        case _CHECK_ATTR_CLASS:
            return 1;
        case _LOAD_ATTR_CLASS_0:
            return 1;
        case _LOAD_ATTR_CLASS_1:
            return 1;
        case _LOAD_ATTR_CLASS:
            return 1;
        case _GUARD_DORV_NO_DICT:
            return 1;
        case _STORE_ATTR_INSTANCE_VALUE:
            return 2;
        case _STORE_ATTR_WITH_HINT:
            return 2;
        case _STORE_ATTR_SLOT:
            return 2;
        case _COMPARE_OP:
            return 2;
        case _COMPARE_OP_FLOAT:
            return 2;
        case _COMPARE_OP_INT:
            return 2;
        case _COMPARE_OP_STR:
            return 2;
        case _IS_OP:
            return 2;
        case _CONTAINS_OP:
            return 2;
        case _CONTAINS_OP_SET:
            return 2;
        case _CONTAINS_OP_DICT:
            return 2;
        case _CHECK_EG_MATCH:
            return 2;
        case _CHECK_EXC_MATCH:
            return 2;
        case _IS_NONE:
            return 1;
        case _GET_LEN:
            return 1;
        case _MATCH_CLASS:
            return 3;
        case _MATCH_MAPPING:
            return 1;
        case _MATCH_SEQUENCE:
            return 1;
        case _MATCH_KEYS:
            return 2;
        case _GET_ITER:
            return 1;
        case _GET_YIELD_FROM_ITER:
            return 1;
        case _FOR_ITER_TIER_TWO:
            return 1;
        case _ITER_CHECK_LIST:
            return 1;
        case _GUARD_NOT_EXHAUSTED_LIST:
            return 1;
        case _ITER_NEXT_LIST:
            return 1;
        case _ITER_CHECK_TUPLE:
            return 1;
        case _GUARD_NOT_EXHAUSTED_TUPLE:
            return 1;
        case _ITER_NEXT_TUPLE:
            return 1;
        case _ITER_CHECK_RANGE:
            return 1;
        case _GUARD_NOT_EXHAUSTED_RANGE:
            return 1;
        case _ITER_NEXT_RANGE:
            return 1;
        case _FOR_ITER_GEN_FRAME:
            return 1;
        case _WITH_EXCEPT_START:
            return 4;
        case _PUSH_EXC_INFO:
            return 1;
        case _GUARD_DORV_VALUES_INST_ATTR_FROM_DICT:
            return 1;
        case _GUARD_KEYS_VERSION:
            return 1;
        case _LOAD_ATTR_METHOD_WITH_VALUES:
            return 1;
        case _LOAD_ATTR_METHOD_NO_DICT:
            return 1;
        case _LOAD_ATTR_NONDESCRIPTOR_WITH_VALUES:
            return 1;
        case _LOAD_ATTR_NONDESCRIPTOR_NO_DICT:
            return 1;
        case _CHECK_ATTR_METHOD_LAZY_DICT:
            return 1;
        case _LOAD_ATTR_METHOD_LAZY_DICT:
            return 1;
        case _CHECK_PERIODIC:
            return 0;
        case _PY_FRAME_GENERAL:
            return 2 + oparg;
        case _CHECK_FUNCTION_VERSION:
            return 2 + oparg;
        case _CHECK_METHOD_VERSION:
            return 2 + oparg;
        case _EXPAND_METHOD:
            return 2 + oparg;
        case _CHECK_IS_NOT_PY_CALLABLE:
            return 2 + oparg;
        case _CALL_NON_PY_GENERAL:
            return 2 + oparg;
        case _CHECK_CALL_BOUND_METHOD_EXACT_ARGS:
            return 2 + oparg;
        case _INIT_CALL_BOUND_METHOD_EXACT_ARGS:
            return 2 + oparg;
        case _CHECK_PEP_523:
            return 0;
        case _CHECK_FUNCTION_EXACT_ARGS:
            return 2 + oparg;
        case _CHECK_STACK_SPACE:
            return 2 + oparg;
        case _INIT_CALL_PY_EXACT_ARGS_0:
            return 2 + oparg;
        case _INIT_CALL_PY_EXACT_ARGS_1:
            return 2 + oparg;
        case _INIT_CALL_PY_EXACT_ARGS_2:
            return 2 + oparg;
        case _INIT_CALL_PY_EXACT_ARGS_3:
            return 2 + oparg;
        case _INIT_CALL_PY_EXACT_ARGS_4:
            return 2 + oparg;
        case _INIT_CALL_PY_EXACT_ARGS:
            return 2 + oparg;
        case _PUSH_FRAME:
            return 1;
        case _CALL_TYPE_1:
            return 3;
        case _CALL_STR_1:
            return 3;
        case _CALL_TUPLE_1:
            return 3;
        case _EXIT_INIT_CHECK:
            return 1;
        case _CALL_BUILTIN_CLASS:
            return 2 + oparg;
        case _CALL_BUILTIN_O:
            return 2 + oparg;
        case _CALL_BUILTIN_FAST:
            return 2 + oparg;
        case _CALL_BUILTIN_FAST_WITH_KEYWORDS:
            return 2 + oparg;
        case _CALL_LEN:
            return 2 + oparg;
        case _CALL_ISINSTANCE:
            return 2 + oparg;
        case _CALL_METHOD_DESCRIPTOR_O:
            return 2 + oparg;
        case _CALL_METHOD_DESCRIPTOR_FAST_WITH_KEYWORDS:
            return 2 + oparg;
        case _CALL_METHOD_DESCRIPTOR_NOARGS:
            return 2 + oparg;
        case _CALL_METHOD_DESCRIPTOR_FAST:
            return 2 + oparg;
        case _MAKE_FUNCTION:
            return 1;
        case _SET_FUNCTION_ATTRIBUTE:
            return 2;
        case _RETURN_GENERATOR:
            return 0;
        case _BUILD_SLICE:
            return 2 + ((oparg == 3) ? 1 : 0);
        case _CONVERT_VALUE:
            return 1;
        case _FORMAT_SIMPLE:
            return 1;
        case _FORMAT_WITH_SPEC:
            return 2;
        case _COPY:
            return 1 + (oparg-1);
        case _BINARY_OP:
            return 2;
        case _SWAP:
            return 2 + (oparg-2);
        case _GUARD_IS_TRUE_POP:
            return 1;
        case _GUARD_IS_FALSE_POP:
            return 1;
        case _GUARD_IS_NONE_POP:
            return 1;
        case _GUARD_IS_NOT_NONE_POP:
            return 1;
        case _JUMP_TO_TOP:
            return 0;
        case _SET_IP:
            return 0;
        case _CHECK_STACK_SPACE_OPERAND:
            return 0;
        case _SAVE_RETURN_OFFSET:
            return 0;
        case _EXIT_TRACE:
            return 0;
        case _CHECK_VALIDITY:
            return 0;
        case _LOAD_CONST_INLINE:
            return 0;
        case _LOAD_CONST_INLINE_BORROW:
            return 0;
        case _POP_TOP_LOAD_CONST_INLINE_BORROW:
            return 1;
        case _LOAD_CONST_INLINE_WITH_NULL:
            return 0;
        case _LOAD_CONST_INLINE_BORROW_WITH_NULL:
            return 0;
        case _CHECK_FUNCTION:
            return 0;
        case _INTERNAL_INCREMENT_OPT_COUNTER:
            return 1;
        case _COLD_EXIT:
            return 0;
        case _DYNAMIC_EXIT:
            return 0;
        case _START_EXECUTOR:
            return 0;
        case _FATAL_ERROR:
            return 0;
        case _CHECK_VALIDITY_AND_SET_IP:
            return 0;
        case _DEOPT:
            return 0;
        case _ERROR_POP_N:
            return oparg;
        case _TIER2_RESUME_CHECK:
            return 0;
        default:
            return -1;
    }
}

#endif // NEED_OPCODE_METADATA


#ifdef __cplusplus
}
#endif
#endif /* !Py_CORE_UOP_METADATA_H */
