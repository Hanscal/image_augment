uthor: "caihua"
date: 2019/5/10
Email: hanscalcai@163.com
'''
import random
import os
import copy
import optparse
import numpy as np
import multiprocessing
import imgaug as ia
import imgaug.augmenters as iaa
from imgaug import multicore
from imgaug.augmentables.batches import UnnormalizedBatch
from PIL import Image, ImageDraw, ImageFont

class genlabel():
    def __init__(self,char_list,opt):
        self.char_list = char_list
        self.opt = opt

    def gen_num(self,num_per_char,num_list=[0,1,2,3,4,5,6,7,8,9]):
        i = 0
        num_total = []
        while i < num_per_char:
            for j in range(len(num_list)):
                # num length
                num_single = []
                # append first non zero num
                num_single.append(random.choice(num_list[1:]))
                num_single.extend(random.sample(num_list,j))
                num_single_str = list(map(str,num_single))
                num_total.append(''.join(num_single_str))
            i+=1
        return num_total

    def add_point(self,num_list,point_position):
        num_list_new = copy.deepcopy(num_list)
        for num in num_list:
            for pos in point_position:
                if len(num)>int(pos) and '.' not in num:
                    num_tmp = num[:-int(pos)]+'.'+num[len(num)-int(pos):]
                    num_list_new.append(num_tmp)
        return num_list_new

    def add_comma(self,num_list):
        num_list_new = copy.deepcopy(num_list)
        for num in num_list:
            num_point_list = num.split('.')
            num_add_temp = []
            num_point_list_reverse = list(num_point_list[0])[::-1]
            for i in range(len(num_point_list_reverse)):
                num_add_temp.append(num_point_list_reverse[i])
                if (i+1)%3==0:
                    num_add_temp.append(',')
            num_add_temp_reverse = num_add_temp[::-1]
            if '.' in num:
                num_list_new.append(''.join(num_add_temp_reverse).lstrip(',')+'.'+num_point_list[-1])
            else:
                num_list_new.append(''.join(num_add_temp_reverse).lstrip(','))
        return num_list_new

    def gen_image(self,input_fonts_dir,label_list,font_list):
        # 返回字典形式
        '''{label:[(filename,image1),(filename,image2)...],label2:[(filename,image1),(filename2,image2)...]}'''
        font_size = 20
        font_color = (0, 0, 0)
        label_image_dict ={}
        for label_id,label in enumerate(label_list):
            image_list = []
            for font_id,font in enumerate(font_list):
                font_path = os.path.join(input_fonts_dir, font)
                if font == '.ttf':
                    print('font name {} is ignored!'.format(font))
                    continue
                try:
                    font = ImageFont.truetype(font_path, font_size)
                    (w, h) = font.getsize(label)
                except Exception as e:
                    print('can not load font: {}, ignored'.format(font_path))
                    continue

                bg_h = int(h * 1.2)
                bg_w = int(w + h)
                bg_img = Image.new("RGB", (bg_w, bg_h), (255, 255, 255))
                draw = ImageDraw.Draw(bg_img)
                draw.text((int(0.5 * h), int(0.08 * h)), label, font=font, fill=font_color)
                file_name = 'label{}_font{}.jpg'.format(str(label_id), str(font_id))
                image_list.append((file_name,bg_img))
                # save_name_list.append(save_name)
            label_image_dict[label] = image_list
        return label_image_dict

    def get_fonts_list(self):
        font_list = os.listdir(self.opt.input_fonts_dir)
        font_list_new = []
        extention = ['ttf','otf']
        for font in font_list:
            if font.split('.')[-1] not in extention:
                continue
            font_list_new.append(font)
        return font_list_new

    def aug_img(self,image_list,aug_ratio):
        sometimes = lambda aug: iaa.Sometimes(aug_ratio, aug)
        seq = iaa.Sequential(
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
        image_list_arr = list(map(np.array, image_list))
        image_list_aug = seq.augment_images(image_list_arr)
        return image_list_aug

    def random_choose(self,label_image_dict,choose_num,choose_ratio):
        """choose the num in the min chosoe_num and choose_ratio*len(image_list)"""
        image_length = sum([len(_) for _ in label_image_dict.values()])
        label_num = len(label_image_dict.keys())
        new_res_dict = {}
        count = 0

        num = min(choose_num,choose_ratio*image_length)
        for label, image_list in label_image_dict.items():
            image_list_new = random.sample(image_list,int(num/label_num))
            new_res_dict[label] = image_list_new
            count += len(image_list_new)
        print('actually choose sample num is {}'.format(count))
        return new_res_dict

    def save_image(self,output_dir,label_save_path,label_image_dict):
        if not os.path.exists(output_dir):
            os.makedir(opt.output_dir)
        fp = open(label_save_path, 'w')
        count = 0
        print('start saving file {}'.format(label_save_path))
        length=sum([len(_) for _ in label_image_dict.values()])
        print('total image num {}'.format(length))
        for label, image_list in label_image_dict.items():
            for img in image_list:
                count+=1
                file_name = img[0]
                bg_img = img[1]
                save_name = os.path.join(opt.output_dir,img[0])
                bg_img.save(save_name)
                fp.write(file_name+'\t'+label+'\n')
                if count%100 ==0:
                    print('saving file {} of {}'.format(count,length))
        fp.close()

def run(opt):
    """img_gen module"""
    num_data = genlabel(char_list=[],opt=opt)
    num_list = num_data.gen_num(num_per_char=100)
    num_list = num_data.add_point(num_list,point_position=[1,2])
    num_list_new = num_data.add_comma(num_list)
    print(len(num_list_new))
    # print(num_list_new)
    font_list = num_data.get_fonts_list()
    label_image_dict = num_data.gen_image(opt.input_fonts_dir, num_list_new,font_list)
    print('total gen images is',sum([len(_) for _ in label_image_dict.values()]))
    label_image_dict_choose = num_data.random_choose(label_image_dict, choose_num=100000, choose_ratio=0.5)
    num_data.save_image(opt.output_dir, './label_font.txt', label_image_dict_choose)
    """"""

    """img_aug module"""
    label_id = 0
    image_aug_dict = {}
    for label, name_image_list in label_image_dict.items():
        label_id += 1
        image_list = [name_image[1] for name_image in name_image_list]
        image_list_aug = num_data.aug_img(image_list,0.2)
        name_image_data = []
        for image_id, image in enumerate(image_list_aug):
            file_name = 'label' + str(label_id) + '_' + 'aug' + str(image_id) + '.jpg'
            image_pil = Image.fromarray(np.uint8(image))
            name_image_data.append((file_name, image_pil))
        image_aug_dict[label] = name_image_data
    # import pdb;pdb.set_trace()
    image_aug_dict_choose = num_data.random_choose(image_aug_dict, choose_num=200000, choose_ratio=0.2)
    num_data.save_image(opt.output_dir, './label_aug.txt', image_aug_dict_choose)
    """"""

def get_options(args=None):
    opt_parser = optparse.OptionParser()
    opt_parser.add_option('-i','--input_fonts_dir',type=str,default='../fonts',help='')
    opt_parser.add_option('-o', '--output_dir', type=str, default='./data', help='')

    (options, args) = opt_parser.parse_args(args=args)
    return options

if __name__=='__main__':
    opt = get_options()
    if not opt:
        os._exit(-1)
    run(opt)




