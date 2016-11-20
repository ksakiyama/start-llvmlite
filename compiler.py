import ast
import sys
import types
import inspect
from collections import OrderedDict
from ctypes import CFUNCTYPE, c_int, c_float

import llvmlite.ir as ir
import llvmlite.binding as llvm


# めんどくさいのでグローバルに宣言
i32 = ir.IntType(32)
f32 = ir.FloatType()

# llvmliteを初期化
llvm.initialize()
llvm.initialize_native_target()
llvm.initialize_native_asmprinter()


def get_ir_type(v):
    """
    引数からLLVM IRの型を返す
    引数はfloatかintの型、floatかintの値
    """
    t = type(v)
    if t == type:
        t = v

    if t == float:
        return f32
    elif t == int:
        return i32
    else:
        print("type: " + str(t))
        raise Exception("ERROR, unknown type")
    return


class TypeInferencer(ast.NodeVisitor):
    """
    型推論する。
    """

    def __init__(self, args):
        self.cnt = 0            # 引数を順番に辿るためのcounter
        self.args = args
        self.nametypes = {}     # 引数の名前とタイプの辞書
        self.ret_type = None

    def generic_visit(self, node):
        return ast.NodeTransformer.generic_visit(self, node)

    def visit_FunctionDef(self, node):
        """
        関数のとき
        """
        # 引数の内容をたどる
        for a in (node.args.args):
            self.visit(a)

        # 関数の内容をたどる
        for b in node.body:
            self.visit(b)

        return node

    def visit_arg(self, node):
        """
        引数のとき
        """
        llvmtype = get_ir_type(self.args[self.cnt])
        # 引数のタイプをセット
        node.type = llvmtype
        # nametypesに登録
        self.nametypes[node.arg] = llvmtype
        self.cnt += 1  # インクリメント

    def visit_Assign(self, node):
        node.value = self.visit(node.value)
        node.targets[0] = self.visit(node.targets[0])

        # valueにはtargetと同じ型を渡す（適当）
        if node.targets[0].type == None:
            # print(str(node.value.id) + ":" + str())
            node.targets[0].type = node.value.type
            self.nametypes[node.targets[0].id] = node.value.type

        # 両方とも型がセットされているはず。。。
        if None == node.value.type and None == node.targets[0].type:
            raise Exception('error : failed type inference')

        # この代入式の型を渡しておく。念の為
        node.type = node.value.type
        # self.ret_type = node.value.type

        return node

    def visit_Name(self, node):
        """
        変数のとき
        """
        # すでにnametypesに登録されている場合
        if node.id in self.nametypes:
            node.type = self.nametypes[node.id]
        else:
            node.type = self.nametypes.get(node.id, None)

        return self.generic_visit(node)

    def visit_Num(self, node):
        """
        数値のとき
        """
        # 値から型を取得してセット
        node.type = get_ir_type(node.n)
        return self.generic_visit(node)

    def visit_Return(self, node):
        """
        returnのとき
        """
        child_node = self.visit(node.value)
        node.type = child_node.type
        self.ret_type = node.value.type
        return node

    def visit_BinOp(self, node):
        """
        四則演算のとき
        """
        node.left = self.visit(node.left)
        node.right = self.visit(node.right)

        # 左右の型があってない場合はエラー
        if not node.left.type.__class__ == node.right.type.__class__:
            raise Exception("ERROR, mismatched type")

        if None == node.left.type and None == node.right.type:
            raise Exception("ERROR, failed type infererence")

        node.type = node.left.type
        return node


class LLVMCodeGenerator(ast.NodeVisitor):
    """
    ASTを作成しLLVM IRへ変換を行う
    """

    def __init__(self, ret_type):
        self.module = ir.Module()
        self.builder = None
        self.fnty = None
        self.func = None
        self.func_name = None

        self.arg_types = []
        self.ret_type = ret_type

        self.arg_nametypes = OrderedDict()  # 引数名とタイプを保存
        self.env_nametypes = {}  # 関数内の変数名を保存。(type, ptr)

    def generic_visit(self, node):
        return ast.NodeTransformer.generic_visit(self, node)

    def visit_FunctionDef(self, node):
        # 引数の内容をたどる
        for a in (node.args.args):
            self.visit(a)

        # 関数の返り値と引数、関数名をセット
        self.fnty = ir.FunctionType(
            self.ret_type, tuple(self.arg_types))
        self.func = ir.Function(self.module, self.fnty, node.name)
        self.func_name = node.name

        # entry
        bb_entry = self.func.append_basic_block()
        self.builder = ir.IRBuilder()
        self.builder.position_at_end(bb_entry)

        # 関数の内容をたどる
        for b in node.body:
            self.visit(b)

    def visit_arg(self, node):
        self.arg_types.append((node.type))
        self.arg_nametypes[node.arg] = node.type

    def visit_Assign(self, node):
        node.targets[0] = self.visit(node.targets[0])
        node.value = self.visit(node.value)

        # 代入先の変数
        if node.targets[0].id in self.env_nametypes:
            # 初登場の変数はallocaしない
            _, ptr_tgt = self.env_nametypes[node.targets[0].id]
        elif node.targets[0].id in self.arg_nametypes:
            # TODO 引数の値に代入
            del self.arg_nametypes[node.targets[0].id]
            ptr_tgt = self.builder.alloca(node.targets[0].type)
            self.env_nametypes[node.targets[0].id] = (
                node.targets[0].type, ptr_tgt)
        else:
            ptr_tgt = self.builder.alloca(node.targets[0].type)
            self.env_nametypes[node.targets[0].id] = (
                node.targets[0].type, ptr_tgt)

        # 代入する変数 or 数値
        if isinstance(node.value, ast.Name):
            # 代入変数が引数にあった場合
            cnt = 0
            if node.value.id in self.arg_nametypes:
                for i in self.arg_nametypes.keys():
                    if node.value.id == i:
                        arg = self.func.args[cnt]
                    cnt += 1
                self.builder.store(arg, ptr_tgt)
            # 代入変数が定義されていれば
            elif node.value.id in self.env_nametypes:
                ptr_val = self.env_nametypes.get[node.value.id]
                self.builder.store(self.builder.load(ptr_val), ptr_tgt)
            # 変数の場合は、どこかで定義されたはず。されてなければエラー

        elif isinstance(node.value, ast.Num):
            self.builder.store(ir.Constant(i32, node.value.n), ptr_tgt)
        else:
            self.builder.store(node.value, ptr_tgt)

        return node

    def visit_Name(self, node):
        return node

    def visit_Num(self, node):
        return node

    def visit_Return(self, node):
        """
        returnのとき
        """
        if isinstance(node.value, ast.Num) or isinstance(node.value, ast.Name):
            val = self.__get_val(node.value)
            self.builder.ret(val)
        elif isinstance(node.value, ast.BinOp):
            return self.builder.ret(self.visit(node.value))
        else:
            raise Exception("ERROR")

    def __get_val(self, node):
        if isinstance(node, ast.Num):
            return ir.Constant(i32, node.n)
        elif isinstance(node, ast.Name):
            cnt = 0
            if node.id in self.arg_nametypes:
                for i in self.arg_nametypes.keys():
                    if node.id == i:
                        return self.func.args[cnt]
                    cnt += 1
            elif node.id in self.env_nametypes:
                _, ptr = self.env_nametypes[node.id]
                return self.builder.load(ptr)
        elif isinstance(node, ast.BinOp):
            return self.builder.ret(self.visit(node.value))
        else:
            return node

    def visit_BinOp(self, node):
        """
        四則演算のとき
        TODO たぶん3つ以上のときにエラー
        """
        node.left = self.visit(node.left)
        node.right = self.visit(node.right)
        val_l = self.__get_val(node.left)
        val_r = self.__get_val(node.right)

        if val_l.type == i32:
            if isinstance(node.op, ast.Add):
                result = self.builder.add(val_l, val_r)
            elif isinstance(node.op, ast.Sub):
                result = self.builder.sub(val_l, val_r)
            elif isinstance(node.op, ast.Mult):
                result = self.builder.mul(val_l, val_r)
            elif isinstance(node.op, ast.Div):
                result = self.builder.sdiv(val_l, val_r)
            elif isinstance(node.op, ast.Mod):
                result = self.builder.srem(val_l, val_r)
            else:
                raise Exception('Unknown operand.')
        elif val_l.type == f32:
            if isinstance(node.op, ast.Add):
                result = self.builder.fadd(val_l, val_r)
            elif isinstance(node.op, ast.Sub):
                result = self.builder.fsub(val_l, val_r)
            elif isinstance(node.op, ast.Mult):
                result = self.builder.fmul(val_l, val_r)
            elif isinstance(node.op, ast.Div):
                result = self.builder.fsdiv(val_l, val_r)
            elif isinstance(node.op, ast.Mod):
                result = self.builder.fsrem(val_l, val_r)
            else:
                raise Exception('Unknown operand.')
        else:
            raise Exception("Unknown data type")

        # self.builder.ret(result)
        return result


class ASTTraverser(ast.NodeVisitor):

    """
    DEBUG用。AST見たいときに使う。
    """

    def generic_visit(self, node):
        print('%12s : %s' % (type(node).__name__, str(node.__dict__)))
        ast.NodeVisitor.generic_visit(self, node)

# end of class ASTTraverser


class Compiler():
    """
    LLVM IRを生成して実行する
    """

    def __init__(self):
        pass

    def exe(self, func, *args):
        """
        第一引数：関数
        第二引数以降：関数に渡す引数
        """
        # 型推論
        typeInf = TypeInferencer(args)
        tree = typeInf.visit(ast.parse(inspect.getsource(func)))

        # print("=======================================")
        # print("== AST ================================")
        # ASTTraverser().visit(tree)
        # print("=======================================")

        # LLVM IRを生成
        codegen = LLVMCodeGenerator(typeInf.ret_type)
        codegen.visit(tree)

        # print("=======================================")
        # print("== LLVM IR ============================")
        # print(codegen.module)
        # print("=======================================")

        # LLVM IR syntaxチェック
        llvm_ir_parsed = llvm.parse_assembly(str(codegen.module))

        # PASSで最適化
        # アホなIRでも最適化してくれる
        pmb = llvm.create_pass_manager_builder()
        pmb.opt_level = 1
        pm = llvm.create_module_pass_manager()
        pmb.populate(pm)
        pm.run(llvm_ir_parsed)

        # JIT
        target_machine = llvm.Target.from_default_triple().create_target_machine()
        engine = llvm.create_mcjit_compiler(llvm_ir_parsed, target_machine)
        engine.finalize_object()

        # ctypesで関数定義
        cfptr = engine.get_function_address(codegen.func_name)

        # 引数の型を設定
        arg_types = []
        for arg in args:
            if type(arg) == int:
                arg_types.append(c_int)
            elif type(arg) == float:
                arg_types.append(c_float)

        cfunc = CFUNCTYPE(*arg_types)(cfptr)

        # 引数の値をキャストする
        arg_values = []
        for arg in args:
            if type(arg) == int:
                arg_values.append(arg)
            elif type(arg) == float:
                arg_values.append(c_float(arg))

        # 実行
        return cfunc(*arg_values)
