import compiler as cp

def thisyear(a, b):
    a = 123
    b = 654
    c = 223
    d = a + b + c
    return d * 2 + 16

def main():
    compiler = cp.Compiler(True)
    ret = compiler(thisyear, 1, 1)
    print(ret)

if __name__ == '__main__':
    main()
