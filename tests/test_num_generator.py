# -*- coding: utf-8 -*-
# email: zhoubingcheng@datagrand.com
# create  : 2019/5/16
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from num_generator import NumGenerator
import unittest


class TestNumGenerator(unittest.TestCase):
    
    def setUp(self):
        self.char_sum = 83214
        self.num_generator = NumGenerator(char_max_len=11, char_sum=83214)
        
    def test_gen_num(self):
        assert (len(list(self.num_generator.gen_num())) == self.char_sum )
        
        
if __name__ == '__main__':
    unittest.main()
    