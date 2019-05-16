"""
author: "caihua"
date: 2019/5/10
Email: hanscalcai@163.com
"""
import os
import optparse
import logging
import json
import codecs
from PIL import Image
from num_generator import NumGenerator
from img_generator import ImageGenerator
logger = logging.getLogger()


def run(opt):
    """img_gen module"""
    num_generator = NumGenerator(char_max_len=11, char_sum=400)
    num_list = list(num_generator.gen_num())
    logger.info('all num len {}'.format(len(num_list)))
    image_generator = ImageGenerator(char_list=num_list, font_dir=opt.input_fonts_dir, cores=4)
    args_img_list = image_generator.gen_images()
    raw_img_list = image_generator.choose_raw_img(args_img_list)
    aug_img_list = image_generator.aug_img(args_img_list)
    logger.info('raw img sum:{}'.format(len(raw_img_list)))
    logging.info('aug img sum:{}'.format(len(aug_img_list)))
    for raw_img_name in image_generator.raw_img_dict:
        raw_img = image_generator.raw_img_dict[raw_img_name].pop('img')
        raw_img_path = os.path.join(opt.output_dir, raw_img_name)
        img = Image.fromarray(raw_img)
        img.save(raw_img_path, quality=100)
    with codecs.open(opt.data_dir, 'w', 'utf-8') as fw:
        fw.write(json.dumps(image_generator.raw_img_dict, ensure_ascii=False, indent=2))
    for i, aug_img in enumerate(aug_img_list):
        aug_img_path = os.path.join(opt.output_dir, '{}_aug.jpg'.format(i))
        img = Image.fromarray(aug_img)
        img.save(aug_img_path, quality=100)
    image_generator.clear()


def get_options(args=None):
    opt_parser = optparse.OptionParser()
    opt_parser.add_option('-i','--input_fonts_dir',type=str,default='/Users/zhoubingcheng/Library/Fonts',help='font dir')
    opt_parser.add_option('-o', '--output_dir', type=str, default='/Users/zhoubingcheng/Desktop/work/testfiles/gen_data/image', help='out image dir')
    opt_parser.add_option('-d', '--data_dir', type=str, default='/Users/zhoubingcheng/Desktop/work/testfiles/gen_data/raw_img.json', help='')
    (options, args) = opt_parser.parse_args(args=args)
    return options

if __name__=='__main__':
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s:[%(levelname)s] %(filename)s %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S')
    opt = get_options()
    if not opt:
        os._exit(-1)
    run(opt)




