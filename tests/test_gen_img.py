# -*- coding: utf-8 -*-
# email: zhoubingcheng@datagrand.com
# create  : 2019/5/16
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import unittest
from img_generator import gen_img
from PIL import Image


class TestGenImg(unittest.TestCase):
    
    def test_gen_img(self):
        font_color = (0, 0, 0)
        font_size = 20
        label = '90980554410'
        font_dir = '/Users/zhoubingcheng/Library/Fonts'
        for i, font_name in enumerate(os.listdir(font_dir)):
            if len(font_name.rsplit('.', 1)) > 1 and font_name.rsplit('.', 1)[1] not \
                    in ['ttf', 'otf'] and not font_name.startswith('.'):
                font_path = os.path.join(font_dir, font_name)
                out_im = gen_img(label, font_color, font_size, font_path)
                img = Image.fromarray(out_im)
                print(i, font_path)
                img.save('test_{}.jpg'.format(i), quality=100)
        

if __name__ == '__main__':
    unittest.main()

