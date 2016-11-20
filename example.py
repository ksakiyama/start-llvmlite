from ctypes import CFUNCTYPE, c_int

import llvmlite.ir as ll
import llvmlite.binding as llvm

"""
これをLLVM IRで作る。
def func(x, y):
    a = x
    b = y
    c = 1000
    d = a
    return a + b + c + d
"""

# 初期化
llvm.initialize()
llvm.initialize_native_target()
llvm.initialize_native_asmprinter()

i32 = ll.IntType(32)

# int func(int, int)
fnty = ll.FunctionType(i32, [i32, i32])
module = ll.Module()
func = ll.Function(module, fnty, name="func")
bb_entry = func.append_basic_block()

builder = ll.IRBuilder()
builder.position_at_end(bb_entry)

# 引数のx, y
x, y = func.args

# 変数a,b,c,dを定義
ptr_a = builder.alloca(i32)
ptr_b = builder.alloca(i32)
ptr_c = builder.alloca(i32)
ptr_d = builder.alloca(i32)

# store
builder.store(x, ptr_a)
builder.store(y, ptr_b)
builder.store(ll.Constant(i32, 1000), ptr_c)

# load
a = builder.load(ptr_a)
b = builder.load(ptr_b)
c = builder.load(ptr_c)

# またstore
builder.store(a, ptr_d)

# 加算して、Returnする
ret1 = builder.add(a, b, name="res")
ret2 = builder.add(ret1, c, name="res2")
ret3 = builder.add(ret2, builder.load(ptr_d), name="res3")
builder.ret(ret3)

llvm_ir = str(module)
llvm_ir_parsed = llvm.parse_assembly(llvm_ir)

print("== LLVM IR ====================")
print(llvm_ir_parsed)

# pass
pmb = llvm.create_pass_manager_builder()
pmb.opt_level = 1
pm = llvm.create_module_pass_manager()
pmb.populate(pm)

pm.run(llvm_ir_parsed)

print("== LLVM IR(opt) ===============")
print(llvm_ir_parsed)

target_machine = llvm.Target.from_default_triple().create_target_machine()

print("== Result =====================")
with llvm.create_mcjit_compiler(llvm_ir_parsed, target_machine) as ee:
    ee.finalize_object()
    cfptr = ee.get_function_address("func")

    cfunc = CFUNCTYPE(c_int, c_int, c_int)(cfptr)
    res = cfunc(100, 2)

    print("res: " + str(res))
