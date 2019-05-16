"""
author: "caihua"
date: 2019/5/10
Email: hanscalcai@163.com
"""

import random
import os
import numpy as np
import multiprocessing
import imgaug as ia
import imgaug.augmenters as iaa
from imgaug import multicore
from imgaug.augmentables.batches import Batch
import functools
from imgaug.augmentables.batches import UnnormalizedBatch
from PIL import Image, ImageDraw, ImageFont
import logging
logger = logging.getLogger()
G = 0


def gen_img(label, font_color, font_size, font_path):
    try:
        font = ImageFont.truetype(font_path, font_size)
        (w, h) = font.getsize(label)
    except Exception as e:
        logger.error('can not load font: {}, ignored'.format(font_path))
        return
    bg_h = int(h * 1.2)
    bg_w = int(w + h)
    bg_img = Image.new("RGB", (bg_w, bg_h), (255, 255, 255))
    draw = ImageDraw.Draw(bg_img)
    draw.text((int(0.5 * h), int(0.08 * h)), label, font=font, fill=font_color)
    img_array = np.array(bg_img).astype(np.uint8)
    return img_array


def gen_img_helper(args):
    return gen_img(*args)


class ImageGenerator(object):
    
    def __init__(self, char_list, raw_choose_num=None, aug_choose_num=None,
                 choose_ratio=0.5, aug_ratio=0.2, font_path_list=[], font_dir=None, cores=4):
        """
        
        :param char_list:
        :param raw_choose_num: choose the num of raw generate images
        :param aug_choose_num: 最终得到的增强图片的数量
        :param aug_ratio:
        :param font_path_list:
        :param font_dir:
        :param cores:
        """
        self.char_list = char_list
        logger.info('prepare char len:{}'.format(len(self.char_list)))
        assert len(font_path_list) > 0 or font_dir is not None
        if font_dir is not None:
            self.font_path_list = [os.path.join(font_dir, font_name) for font_name in os.listdir(font_dir)  \
                                   if len(font_name.rsplit('.', 1)) > 1 and font_name.rsplit('.', 1)[1] not \
                                   in ['ttf', 'otf'] and not font_name.startswith('.')]
        else:
            self.font_path_list = font_path_list
        logger.info('font sum:{}'.format(len(self.font_path_list)))
        logger.info('prepare generate img sum:{}'.format(len(self.char_list) * len(self.font_path_list)))
        self.raw_choose_num = raw_choose_num
        self.aug_choose_num = aug_choose_num
        self.choose_ratio = choose_ratio
        self.aug_ratio = aug_ratio
        self.cores = cores
        sometimes = lambda aug: iaa.Sometimes(aug_ratio, aug)
        self.seq = iaa.Sequential(
            [
                sometimes(iaa.CropAndPad(
                    percent=(-0.05, 0.1),
                    pad_mode=ia.ALL,
                    pad_cval=(0, 255)
                )),
                sometimes(iaa.Affine(
                    scale={"x": (0.8, 1.2), "y": (0.8, 1.2)},
                    translate_percent={"x": (-0.1, 0.1), "y": (-0.05, 0.05)},
                    rotate=(-5, 5),
                    shear=(-3, 3),
                    order=[0, 1],
                    cval=(0, 255),
                    mode=ia.ALL
                )),
                # execute 0 to 5 of the following augmenters per image
                iaa.SomeOf((0, 5),
                           [
                               iaa.OneOf([
                                   iaa.GaussianBlur((0, 0.5)),
                                   iaa.AverageBlur(k=(1, 3)),
                                   iaa.MedianBlur(k=(1, 3)),
                                   iaa.Dropout(p=0.05),
                                   iaa.CoarseDropout(p=0.05, size_percent=0.20),
                                   iaa.Salt(p=0.03),
                                   iaa.GammaContrast(gamma=0.81),
                                   iaa.Emboss(alpha=1, strength=0.5)
                               ]),
                           ],
                           random_order=True
                           )
            ],
            random_order=True
        )
        self.raw_img_dict = {}
        self.aug_imgs = []

    def clear(self):
        self.raw_img_dict = {}
        self.aug_imgs = []

    def gen_images(self):
        """
        :return: 参数列表，形成的图片列表
        """
        font_size = 20
        font_color = (0, 0, 0)
        args_list = []
        for font_id, font_path in enumerate(self.font_path_list):
            for label_id, label in enumerate(self.char_list):
                args_list.append([label, font_color, font_size, font_path])
        logger.info('begin generate bg img')
        if self.cores <= 1:
            bg_imgs = [gen_img(*arg) for arg in args_list]
        else:
            pool = multiprocessing.Pool(processes=int(self.cores))
            bg_imgs = pool.map(gen_img_helper, args_list)
            pool.close()
            pool.join()
        logger.info('success generate all bg img')
        return [args_list[i] + [bg_imgs[i]] for i in range(len(args_list))]
   
    def choose_raw_img(self, args_img_list):
        valid_args_img_list = [args_img for args_img in args_img_list if args_img[-1] is not None]
        raw_choose_num = int(len(args_img_list) * self.choose_ratio) if self.raw_choose_num is None \
            else min(self.raw_choose_num, len(args_img_list) * self.choose_ratio)
        logger.info('choose raw img sum:{}'.format(raw_choose_num))
        raw_args_img_list = random.sample(valid_args_img_list, raw_choose_num)
        raw_imgs_list = []
        for i, raw_args_img in enumerate(raw_args_img_list):
            label, font_color, font_size, font_path, img = raw_args_img
            img_name = '{}_raw.jpg'.format(i)
            self.raw_img_dict[img_name] = {"img": img, "label": label, "font_size": font_size,
                                  "font_color": font_color, "font_path": font_path}
            raw_imgs_list.append(img)
        return raw_imgs_list
 
    def aug_img(self, args_img_list):
        logger.info('begin generate aug img')
        if self.cores <= 1:
            img_list = [args_img[-1] for args_img in args_img_list if args_img[-1] is not None]
            image_aug_list = self.seq.augment_images(img_list)
        else:
            img_args_list_copy = args_img_list.copy()
            char_sum = len(self.char_list)
            c = 0
            batches = []
            batch_imgs = []
            while img_args_list_copy:
                label, font_color, font_size, font_path, img = img_args_list_copy.pop(0)
                if img is not None:
                    batch_imgs.append(img)
                c += 1
                if c == char_sum:
                    if len(batch_imgs) > 0:
                        batch = Batch(batch_imgs)
                        batches.append(batch)
                    c = 0
                    batch_imgs = []
            seq_pool = multicore.Pool(self.seq, processes=self.cores)
            gen_img_batches = seq_pool.map_batches(batches, chunksize=4)
            seq_pool.close()
            seq_pool.join()
            image_aug_list = []
            for gen_img_batch in gen_img_batches:
                image_aug_list.extend(gen_img_batch.images_unaug)
        logger.info('success generate aug img')
        return image_aug_list
