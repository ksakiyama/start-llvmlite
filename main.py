import compiler as cp
import inspect
import types


def add(a, b):
    return a + b


def main():
    compiler = cp.Compiler()
    ret = compiler.exe(add, 1, 1)
    print(ret)

if __name__ == '__main__':
    main()
