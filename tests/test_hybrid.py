import sys
from pathlib import Path
import unittest
sys.path.insert(0, str(Path('__file__').resolve().parent))

from lyra.__core.cmath import factorial_c
from lyra.__core.descent import factorial_cpp, vectorize, optimize

class MainTest(unittest.TestCase):
    def test_c(self):
        self.assertEqual(40320, factorial_c(8))
    def test_cpp(self):
        self.assertEqual(40320, factorial_cpp(8))

if __name__ == '__main__':
    unittest.main()
