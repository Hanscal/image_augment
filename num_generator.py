# -*- coding: utf-8 -*-
# email: zhoubingcheng@datagrand.com
# create  : 2019/5/16
import random
import math


class NumGenerator(object):
    
    def __init__(self, char_max_len, char_sum, add_comma=True, add_point=True):
        assert char_max_len >= 1
        self.char_max_len = int(char_max_len)
        self.char_sum = int(char_sum)
        self.add_comma = add_comma
        self.add_point = add_point
    
    def gen_single_num(self, char_len):
        num_str = str(random.randint(10 ** (char_len - 1), 10 ** char_len - 1))
        if self.add_point:
            position = random.choice([0, 1, 2])
            if len(num_str) > position > 0:
                num_str = num_str[:-position] + '.' + num_str[-position:]
        if self.add_comma and random.choice([True, False]):
            if len(num_str.split('.', 1)) == 1:
                int_str = num_str
                dec_str = ''
            else:
                int_str, dec_str = num_str.split('.', 1)
                dec_str = '.' + dec_str
            if len(int_str) > 3:
                reverse_str = int_str[::-1]
                comma_len = math.ceil(((len(reverse_str) - 1) // 3))
                new_reverse_int_str = ','.join([reverse_str[3 * i: 3 * i + 3] for i in range(comma_len + 1)])
                int_str = new_reverse_int_str[::-1]
            num_str = int_str + dec_str
        return num_str
    
    def gen_num(self):
        for i in range(1, self.char_max_len + 1):
            if i == self.char_max_len:
                char_sum = self.char_sum - self.char_sum // self.char_max_len * (self.char_max_len - 1)
            else:
                char_sum = self.char_sum // self.char_max_len
            for j in range(char_sum):
                yield self.gen_single_num(i)