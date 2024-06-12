import os.path
import torch
import torchvision.transforms.functional as tf
import torch.nn.functional as F
from PIL import Image, ImageStat
import numpy as np
import torchvision.transforms as transforms
import random
import cv2
import matplotlib.pyplot as plt
import glob
from skimage.metrics import mean_squared_error as mse
import argparse
from tqdm import tqdm

parser = argparse.ArgumentParser(description='some parameters about temporal consistency evaluation.')

parser.add_argument('--dataset_root', type=str, default='../datasets/HYouTube/', help='path of dataset')
parser.add_argument('--experiment_name', type=str, default='swin3d_skipnorm_patch_v2_1H6L_win4_p1_LT_L2', help='folder name in the results folder')
parser.add_argument('--mode', type=str, default='v', help='v, rgb, gray or hsv')
# parser.add_argument('--metric', type=str, default='fR-RTC', help='fR-RTC, R-RTC, fRTC or RTC')
parser.add_argument('--brightness_region', type=str, default='foreground', help='forground or image')
parser.add_argument('--plot', default=False, action="store_true", help='whether plot or not')
parser.add_argument('--plot_bar_gradient_diff', default=False, action="store_true", help='whether plot or not')

args = parser.parse_args()

def extract_image_brightness(args, frame_path, mask_path):
    im = cv2.imread(frame_path)
    if args.mode == 'rgb':
        v = cv2.cvtColor(im,cv2.COLOR_BGR2RGB)
    elif args.mode == 'hsv':
        v = cv2.cvtColor(im, cv2.COLOR_BGR2HSV)  # [256, 256, 3]
    elif args.mode == 'v':
        hsv_img = cv2.cvtColor(im, cv2.COLOR_BGR2HSV)  # [256, 256, 3]
        v = cv2.split(hsv_img)[2]  # [256, 256]
    elif args.mode == 'gray':
        v = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    else:
        raise NotImplementedError('Mode Error')
    
    v_mean = np.mean(v)
    return v_mean
    # return v.astype(np.float32)


def extract_brightness_with_hsv(args, frame_path, mask_path):
    im = cv2.imread(frame_path)
    if args.mode == 'rgb':
        v = cv2.cvtColor(im,cv2.COLOR_BGR2RGB)
        mask = np.tile(mask.reshape(256,256,1),(1,1,3))
    elif args.mode == 'hsv':
        v = cv2.cvtColor(im, cv2.COLOR_BGR2HSV)  # [256, 256, 3]
        mask = np.tile(mask.reshape(256,256,1),(1,1,3))
    elif args.mode == 'v':
        hsv_img = cv2.cvtColor(im, cv2.COLOR_BGR2HSV)  # [256, 256, 3]
        v = cv2.split(hsv_img)[2]  # [256, 256]
        mask = np.array(Image.open(mask_path).convert('1').resize((256,256)))
    elif args.mode == 'gray':
        v = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        mask = np.array(Image.open(mask_path).convert('1').resize((256,256)))
    else:
        raise NotImplementedError('Mode Error')

    fg_v = v * mask
    fg_area = np.sum(mask)
    fg_v_mean = np.sum(fg_v) / fg_area
    return fg_v_mean

def plot_brightness(args, bright_values_dict, save_name, info_root):
    frame = [str(i) for i in range(1,21)]
    plt.figure()
    # plt.plot(frame, diff, label = 'diff')
    for video_cls, bright_values in bright_values_dict.items():
        plt.plot(frame, bright_values, label = video_cls, marker='o')
    plt.legend()
    plot_save_path = 'plot/' + args.mode + '_value'
    plot_save_path = os.path.join(info_root, plot_save_path)
    if not os.path.exists(plot_save_path):
        os.makedirs(plot_save_path)
    save_path = os.path.join(plot_save_path, save_name.replace('/','_') + '.png')
    plt.savefig(save_path)
    plt.close()

def plot_gradient(args, bright_gradients_dict, save_name, info_root):
    frame = [str(i) for i in range(1,20)]
    plt.figure()
    for video_cls, bright_gradients in bright_gradients_dict.items():
        plt.plot(frame, bright_gradients, label = video_cls, marker='o')
    plt.legend()
    plot_save_path = 'plot/' + args.mode + '_gradient'
    plot_save_path = os.path.join(info_root, plot_save_path)
    if not os.path.exists(plot_save_path):
        os.makedirs(plot_save_path)
    save_path = os.path.join(plot_save_path, save_name.replace('/','_') + '.png')
    plt.savefig(save_path)
    plt.close()
    
def plot_bar_gradient_diff(args, harm_real_diff, mu, save_name, info_root):
    frame = [str(i) for i in range(1,20)]
    plt.figure()
    plt.bar(frame, harm_real_diff)
    # plt.bar(frame, harm_real_diff, reverse=True)
    plt.plot(frame, [mu for i in range(1,20)], '--',color='r')
    plt.xticks([index - 0.4 for index in range(0,20)], [index for index in range(0,20)])
    plt.xlabel("frame")
    plt.ylabel("gradient difference")
    # plt.title(save_name)
    plot_save_path = 'plot/' + 'bar_gradient_diff_' + args.mode
    plot_save_path = os.path.join(info_root, plot_save_path)
    if not os.path.exists(plot_save_path):
        os.makedirs(plot_save_path)
    save_path = os.path.join(plot_save_path, save_name.replace('/','_') + '.png')
    plt.savefig(save_path)
    plt.close()

testfile = args.dataset_root+'test_list.txt'
video_objs = []
with open(testfile, 'r') as f:
    for line in f.readlines():
        video_objs.append(line.rstrip().split(' ')[-1].replace('synthetic_composite_videos/', ''))  # ['ff692fdc56/object_1', 'ff773b1a1e/object_0']


count = 0
gradient_mse_scores = 0.0
gradient_mse_score_list = []
relation_scores = 0.0
relation_score_list = []

results_root = 'results/' + args.experiment_name + '/test_latest/images/'
info_root = results_root.replace('images','info')
if not os.path.exists(info_root):
    os.makedirs(info_root)

for i,video_obj in enumerate(tqdm(video_objs)):
    video_obj_path = results_root + video_obj

    frame_paths_dict = {}
    frame_paths_dict['real'] = glob.glob(os.path.join(video_obj_path, '*real*'))
    frame_paths_dict['harmonized'] = glob.glob(os.path.join(video_obj_path, '*harm*'))
    for value in frame_paths_dict.values():
        value.sort()
    
    
    videos_cls = ['real', 'harmonized']
    bright_gradients_dict = {}
    bright_values_dict = {}
    for video_cls in videos_cls:
        bright_values_dict[video_cls] = []
    for video_cls, frame_paths in frame_paths_dict.items():
        for frame_path in frame_paths:
            frame_index = frame_path.split('/')[-1].replace((video_obj.replace('/','_')+'_'),'')[:5]
            mask_path = args.dataset_root + 'foreground_mask/' + video_obj + '/' + frame_index + '.png'

            if args.brightness_region == 'image':
                bright_value = extract_image_brightness(args, frame_path, mask_path)
            elif args.brightness_region == 'foreground':
                bright_value = extract_brightness_with_hsv(args, frame_path, mask_path)
            else:
                raise NotImplementedError('Brightness Region Error')
            
            bright_values_dict[video_cls].append(bright_value)
        bright_gradients_dict[video_cls] = np.array(bright_values_dict[video_cls][1:]) - np.array(bright_values_dict[video_cls][:-1])
        
    if args.plot:
        plot_brightness(args, bright_values_dict, video_obj, info_root)
        plot_gradient(args, bright_gradients_dict, video_obj, info_root)

    # break
    count += 1
    
    mse_score = mse(bright_gradients_dict['real'], bright_gradients_dict['harmonized'])
    harm_real_diff = np.abs(bright_gradients_dict['harmonized']-bright_gradients_dict['real'])
    mu = np.mean(harm_real_diff)
    gradient_mse_score = mse_score + np.sum((np.array(harm_real_diff)[harm_real_diff>mu] - mu)**2) / len(harm_real_diff)
    gradient_mse_scores += gradient_mse_score
    gradient_mse_info = (video_obj, gradient_mse_score)
    gradient_mse_score_list.append(round(gradient_mse_score,2))

    relation_scores += mse_score
    relation_info = (video_obj, mse_score)
    relation_score_list.append(round(mse_score,2))


    if args.plot_bar_gradient_diff:
        plot_bar_gradient_diff(args, harm_real_diff, mu, video_obj, info_root)

gradient_mse_score_mu = gradient_mse_scores / count
save_gradient_mse_path = 'R-RTC' + '_' + args.mode + '.txt'
if args.brightness_region == 'foreground':
    save_gradient_mse_path = 'f' + save_gradient_mse_path
save_gradient_mse_path = os.path.join(info_root, save_gradient_mse_path)
file=open(save_gradient_mse_path,'w')
file.write(str(round(gradient_mse_score_mu, 2)))
file.write('\n')
for line in gradient_mse_score_list:
    file.write(str(line)+"\n")
file.close() 

if args.brightness_region == 'foreground':
    print('fR-RTC:',round(gradient_mse_score_mu, 2))
else:
    print('R-RTC:',round(gradient_mse_score_mu, 2))

relation_score_mu = relation_scores / count
save_relation_path = 'RTC' + '_' + args.mode + '.txt'
if args.brightness_region == 'foreground':
    save_relation_path = 'f' + save_relation_path
save_relation_path = os.path.join(info_root, save_relation_path)
file=open(save_relation_path,'w')
file.write(str(round(relation_score_mu, 2)))
file.write('\n')
for line in relation_score_list:
    file.write(str(line)+"\n")
file.close() 

if args.brightness_region == 'foreground':
    print('fRTC:',round(relation_score_mu, 2))
else:
    print('RTC:',round(relation_score_mu, 2))