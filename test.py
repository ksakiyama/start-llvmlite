import unittest
import compiler as cp
import random
import numpy as np


def add(a, b):
    return a + b


def sub(a, b):
    return a - b


def mul(a, b):
    return a * b


def div(a, b):
    return a / b


def rem(a, b):
    return a % b


def mul(a, b):
    return a * b


def add2(a, b):
    x = a
    return x + b


def add3(a, b):
    x = 100
    y = 100
    return x + y


def add4(a, b, c):
    x = a + b
    return x + c


def add5(a, b, c):
    return a + b + c


def add6(a, b, c, d):
    return a + b + c + d


def add7(a, b):
    x = a + 1
    b = 200
    return x + b


def mod(a, b, c):
    return a * b + c


class UnitTester(unittest.TestCase):

    def setUp(self):
        self.compiler = cp.Compiler()

    def test_add(self):
        for _ in range(100):
            x = random.randint(-100, 100)
            y = random.randint(-100, 100)
            ret = self.compiler.exe(add, x, y)
            self.assertEqual(x + y, ret)

    def test_sub(self):
        for _ in range(100):
            x = random.randint(-100, 100)
            y = random.randint(-100, 100)
            ret = self.compiler.exe(sub, x, y)
            self.assertEqual(x - y, ret)

    def test_mul(self):
        for _ in range(100):
            x = random.randint(-100, 100)
            y = random.randint(-100, 100)
            ret = self.compiler.exe(mul, x, y)
            self.assertEqual(x * y, ret)

    def test_div(self):
        for _ in range(100):
            x = random.randint(-100, 100)
            y = random.randint(1, 100)
            ret = self.compiler.exe(div, x, y)
            self.assertEqual(int(x / y), ret)

    def test_rem(self):
        for _ in range(100):
            x = random.randint(0, 100)
            y = random.randint(1, 100)
            ret = self.compiler.exe(rem, x, y)
            self.assertEqual(x % y, ret)

    def test_add_float(self):
        x = 1.1
        y = 2.1
        ret = self.compiler.exe(add, x, y)
        self.assertEqual(np.float32(x) + np.float32(y), ret)

    def test_mul_float(self):
        x = 1.1
        y = 2.1
        ret = self.compiler.exe(mul, x, y)
        self.assertEqual(np.float32(x) * np.float32(y), ret)

    def test_add2(self):
        x = 10
        y = 20
        ret = self.compiler.exe(add2, x, y)
        self.assertEqual(x + y, ret)

    def test_add3(self):
        ret = self.compiler.exe(add3, 1, 1)
        self.assertEqual(200, ret)

    def test_add4(self):
        x = 100
        y = 200
        z = -1
        ret = self.compiler.exe(add4, x, y, z)
        self.assertEqual(add4(x, y, z), ret)

    def test_add5(self):
        x = 100
        y = 200
        z = -1
        ret = self.compiler.exe(add5, x, y, z)
        self.assertEqual(add5(x, y, z), ret)

    def test_add6(self):
        x = 100
        y = 200
        z = -1
        w = 999
        ret = self.compiler.exe(add6, x, y, z, w)
        self.assertEqual(add6(x, y, z, w), ret)

    def test_add7(self):
        x = 1
        y = 1
        ret = self.compiler.exe(add7, x, y)
        self.assertEqual(add7(x, y), ret)

    def test_mod(self):
        x = 1
        y = 1
        z = 1
        ret = self.compiler.exe(mod, x, y, z)
        self.assertEqual(mod(x, y, z), ret)

    def test_mod_float(self):
        x = 2.0
        y = 1.1
        z = 9.0
        ret = self.compiler.exe(mod, x, y, z)
        self.assertEqual(mod(np.float32(x), np.float32(y), np.float32(z)), ret)


if __name__ == '__main__':
    unittest.main()
