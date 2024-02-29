import os
import cv2
import sys
import pdb
import six
import glob
import time
import torch
import json
import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)

import numpy as np
# import pyarrow as pa
from PIL import Image
import torch.utils.data as data
from utils import video_augmentation
from torch.utils.data.sampler import Sampler

sys.path.append("..")


class ph14(data.Dataset):
    def __init__(self, prefix, gloss_dict, drop_ratio=1, num_gloss=-1, mode="train", transform_mode=True,
                 datatype="lmdb"):
        self.mode = mode
        self.ng = num_gloss
        self.prefix = prefix
        self.dict = gloss_dict
        self.data_type = datatype
        self.feat_prefix = f"{prefix}/features/fullFrame-256x256px/{mode}"
        self.transform_mode = "train" if transform_mode else "test"
        self.inputs_list = np.load(f"./preprocess/phoenix2014/{mode}_info.npy", allow_pickle=True).item()
        # self.inputs_list = np.load(f"{prefix}/annotations/manual/{mode}.corpus.npy", allow_pickle=True).item()
        # self.inputs_list = np.load(f"{prefix}/annotations/manual/{mode}.corpus.npy", allow_pickle=True).item()
        # self.inputs_list = dict([*filter(lambda x: isinstance(x[0], str) or x[0] < 10, self.inputs_list.items())])
        print(mode, len(self))
        self.data_aug = self.transform()
        print("")

    def __getitem__(self, idx):
        if self.data_type == "video":
            input_data, skeleton, label = self.read_video(idx)
            input_data, skeleton, label = self.normalize(input_data, skeleton, label)
            return input_data, skeleton, label, self.inputs_list[idx]['original_info']


    def read_video(self, index, num_glosses=-1):
        # load file info
        fi = self.inputs_list[index]
        img_folder = os.path.join(self.prefix, "features/fullFrame-256x256px/" + fi['folder'])
        img_list = sorted(glob.glob(img_folder))
        ske_folder = os.path.join(self.prefix, "features/skeleton/" + fi['folder']).replace('.png','.json')
        ske_list = sorted(glob.glob(ske_folder))
        ske_series = []
        pose_keypoints_slice = np.array([0, 4, 7])
        for skeleton_path in ske_list:
            with open(skeleton_path,'r') as load_f:
                load_dict = json.load(load_f)
                people = load_dict['people'][0]
                pose_keypoints_2d = people['pose_keypoints_2d']

                pose_keypoints = []
                for i in range(0, len(pose_keypoints_2d), 3):
                    pose_keypoints.append(pose_keypoints_2d[i: i+3])
                pose_keypoints = np.array(pose_keypoints)
                pose_keypoints = pose_keypoints[pose_keypoints_slice]
                ske_series.append(pose_keypoints)
                
        skeleton = np.stack(ske_series, axis = 0)
        skeleton[:, :, 0] = skeleton[:, :, 0] * (256 / 210)
        skeleton[:, :, 1] = skeleton[:, :, 1] * (256 / 260)
        
        label_list = []
        for phase in fi['label'].split(" "):
            if phase == '':
                continue
            if phase in self.dict.keys():
                label_list.append(self.dict[phase][0])
        return [cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB) for img_path in img_list], skeleton, torch.LongTensor(label_list)
    
    def read_features(self, index):
        # load file info
        fi = self.inputs_list[index]
        data = np.load(f"./features/{self.mode}/{fi['fileid']}_features.npy", allow_pickle=True).item()
        return data['features'], data['label']

    def normalize(self, video, skeleton, label, file_id=None):
        video, skeleton, label = self.data_aug(video, skeleton, label, file_id)
        video = video.float() / 127.5 - 1
        return video, skeleton, label

    def transform(self):
        if self.transform_mode == "train":
            print("Apply training transform.")
            return video_augmentation.Compose([
                # video_augmentation.CenterCrop(224),
                # video_augmentation.WERAugment('/lustre/wangtao/current_exp/exp/baseline/boundary.npy'),
                video_augmentation.RandomHorizontalFlip_ske(0.5),
                video_augmentation.RandomCrop_ske(224),
                #video_augmentation.resize(224),
                video_augmentation.ToTensor(),
                video_augmentation.TemporalRescale(0.2),
                # video_augmentation.Resize(0.5),
            ])
        else:
            print("Apply testing transform.")
            return video_augmentation.Compose([
                video_augmentation.CenterCrop_ske(224),
                #video_augmentation.Resize(0.5),
                #video_augmentation.resize(224),
                video_augmentation.ToTensor(),
            ])

    def byte_to_img(self, byteflow):
        unpacked = pa.deserialize(byteflow)
        imgbuf = unpacked[0]
        buf = six.BytesIO()
        buf.write(imgbuf)
        buf.seek(0)
        img = Image.open(buf).convert('RGB')
        return img

    @staticmethod
    def collate_fn(batch):
        batch = [item for item in sorted(batch, key=lambda x: len(x[0]), reverse=True)]
        video, skeleton, label, info = list(zip(*batch))
        # padding video
        max_len_v = len(video[0])
        video_length = torch.LongTensor([len(vid) for vid in video])
        padded_video = torch.stack([torch.cat([vid, torch.zeros(vid[0].shape).expand(max_len_v - len(vid), *vid[0].shape)]) for vid in video], axis = 0)

        # padding skeleton
        max_len_s = len(skeleton[0])
        padded_skeleton = torch.stack([torch.cat([ske, torch.zeros(ske[0].shape).expand(max_len_s - len(ske), *ske[0].shape)]) for ske in skeleton], axis = 0)

        #padding label
        max_len_l = max([len(l) for l in label])
        label_length = torch.LongTensor([len(lab) for lab in label])
        padded_label = torch.stack([torch.cat([lab, torch.LongTensor([0] * (max_len_l - len(lab)))])  for lab in label], axis = 0)
        return padded_video, video_length, padded_skeleton, padded_label, label_length, info

    def __len__(self):
        return len(self.inputs_list) - 1

    def record_time(self):
        self.cur_time = time.time()
        return self.cur_time

    def split_time(self):
        split_time = time.time() - self.cur_time
        self.record_time()
        return split_time

class CSLFeeder(data.Dataset):
    def __init__(self, prefix, gloss_dict, drop_ratio=1, num_gloss=-1, mode="train", transform_mode=True):
        self.mode = mode
        self.ng = num_gloss
        self.prefix = prefix
        self.dict = gloss_dict
        self.transform_mode = "train" if transform_mode else "test"
        self.inputs_list = np.load(f"./preprocess/CSL/{mode}_info.npy", allow_pickle=True).item()
        print(mode, len(self))
        self.data_aug = self.transform()
        print("")

    def __getitem__(self, idx):
        input_data, skeleton, label = self.read_video(idx)
        input_data, skeleton, label = self.normalize(input_data, skeleton, label)
        return input_data, skeleton, label, self.inputs_list[idx]['fileid']


    def read_video(self, index, num_glosses=-1):
        # load file info
        fi = self.inputs_list[index]
        img_folder = os.path.join(self.prefix, 'ccslSub_resized', fi['folder'], '*.png')
        img_list = sorted(glob.glob(img_folder))
        ske_folder = os.path.join(self.prefix, 'ccslskedataSub', fi['folder'], '*.json')
        ske_list = sorted(glob.glob(ske_folder))
        ske_series = []
        pose_keypoints_slice = np.array([0, 4, 7])
        for skeleton_path in ske_list:
            with open(skeleton_path,'r') as load_f:
                load_dict = json.load(load_f)
                people = load_dict['people'][0]
                pose_keypoints_2d = people['pose_keypoints_2d']

                pose_keypoints = []
                for i in range(0, len(pose_keypoints_2d), 3):
                    pose_keypoints.append(pose_keypoints_2d[i: i+3])
                pose_keypoints = np.array(pose_keypoints)
                pose_keypoints = pose_keypoints[pose_keypoints_slice]
                ske_series.append(pose_keypoints)
                
        skeleton = np.stack(ske_series, axis = 0)
        skeleton[:, :, 0] = skeleton[:, :, 0] * (256 / 640)
        skeleton[:, :, 1] = skeleton[:, :, 1] * (256 / 460)
        
        label_list = []
        for phase in fi['label'].split(" "):
            if phase == '':
                continue
            if phase in self.dict.keys():
                label_list.append(self.dict[phase][0])
        return [cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB) for img_path in img_list], skeleton, torch.LongTensor(label_list)
    

    def normalize(self, video, skeleton, label, file_id=None):
        video, skeleton, label = self.data_aug(video, skeleton, label, file_id)
        video = video.float() / 127.5 - 1
        return video, skeleton, label

    def transform(self):
        if self.transform_mode == "train":
            print("Apply training transform.")
            return video_augmentation.Compose([
                # video_augmentation.CenterCrop(224),
                # video_augmentation.WERAugment('/lustre/wangtao/current_exp/exp/baseline/boundary.npy'),
                video_augmentation.RandomHorizontalFlip_ske(0.5),
                video_augmentation.RandomCrop_ske(224),
                #video_augmentation.resize(224),
                video_augmentation.ToTensor(),
                video_augmentation.TemporalRescale(0.2),
                # video_augmentation.Resize(0.5),
            ])
        else:
            print("Apply testing transform.")
            return video_augmentation.Compose([
                video_augmentation.CenterCrop_ske(224),
                #video_augmentation.Resize(0.5),
                #video_augmentation.resize(224),
                video_augmentation.ToTensor(),
            ])

    def byte_to_img(self, byteflow):
        unpacked = pa.deserialize(byteflow)
        imgbuf = unpacked[0]
        buf = six.BytesIO()
        buf.write(imgbuf)
        buf.seek(0)
        img = Image.open(buf).convert('RGB')
        return img

    @staticmethod
    def collate_fn(batch):
        batch = [item for item in sorted(batch, key=lambda x: len(x[0]), reverse=True)]
        video, skeleton, label, info = list(zip(*batch))
        # padding video
        max_len_v = len(video[0])
        video_length = torch.LongTensor([len(vid) for vid in video])
        padded_video = torch.stack([torch.cat([vid, torch.zeros(vid[0].shape).expand(max_len_v - len(vid), *vid[0].shape)]) for vid in video], axis = 0)

        # padding skeleton
        max_len_s = len(skeleton[0])
        padded_skeleton = torch.stack([torch.cat([ske, torch.zeros(ske[0].shape).expand(max_len_s - len(ske), *ske[0].shape)]) for ske in skeleton], axis = 0)

        #padding label
        max_len_l = max([len(l) for l in label])
        label_length = torch.LongTensor([len(lab) for lab in label])
        padded_label = torch.stack([torch.cat([lab, torch.LongTensor([0] * (max_len_l - len(lab)))])  for lab in label], axis = 0)
        return padded_video, video_length, padded_skeleton, padded_label, label_length, info

    def __len__(self):
        return len(self.inputs_list) - 1

    def record_time(self):
        self.cur_time = time.time()
        return self.cur_time

    def split_time(self):
        split_time = time.time() - self.cur_time
        self.record_time()
        return split_time


class ph14t(data.Dataset):
    def __init__(self, prefix, gloss_dict, mode="train", transform_mode=True):
        self.mode = mode.split('-')[0]
        self.prefix = prefix
        self.dict = gloss_dict
        self.feat_prefix = f"{prefix}/features/fullFrame-256x256px/{mode}"
        self.transform_mode = "train" if transform_mode else "test"
        self.inputs_list = np.load(f"./preprocess/PHOENIX-2014-T/{mode}_info.npy", allow_pickle=True).item()
        print(mode, len(self))
        self.data_aug = self.transform()
        print("")

    def __getitem__(self, idx):
        input_data, skeleton, label = self.read_video(idx)
        input_data, skeleton, label = self.normalize(input_data, skeleton, label)
        return input_data, skeleton, label, self.inputs_list[idx]['original_info']


    def read_video(self, index, num_glosses=-1):
        # load file info
        fi = self.inputs_list[index]
        img_folder = os.path.join(self.prefix, "features/fullFrame-256x256px/" + fi['folder'])
        img_list = sorted(glob.glob(img_folder))
        ske_folder = os.path.join(self.prefix, "features/skeleton/" + fi['folder']).replace('.png','.json')
        ske_list = sorted(glob.glob(ske_folder))
        ske_series = []
        pose_keypoints_slice = np.array([0, 4, 7])
        for skeleton_path in ske_list:
            with open(skeleton_path,'r') as load_f:
                load_dict = json.load(load_f)
                people = load_dict['people'][0]
                pose_keypoints_2d = people['pose_keypoints_2d']

                pose_keypoints = []
                for i in range(0, len(pose_keypoints_2d), 3):
                    pose_keypoints.append(pose_keypoints_2d[i: i+3])
                pose_keypoints = np.array(pose_keypoints)
                pose_keypoints = pose_keypoints[pose_keypoints_slice]
                ske_series.append(pose_keypoints)
                
        skeleton = np.stack(ske_series, axis = 0)
        skeleton[:, :, 0] = skeleton[:, :, 0] * (256 / 210)
        skeleton[:, :, 1] = skeleton[:, :, 1] * (256 / 260)
        
        label_list = []
        for phase in fi['label'].split(" "):
            if phase == '':
                continue
            if phase in self.dict.keys():
                label_list.append(self.dict[phase][0])
                
        return [cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB) for img_path in img_list], skeleton, torch.LongTensor(label_list)
    

    def normalize(self, video, skeleton, label, file_id=None):
        video, skeleton, label = self.data_aug(video, skeleton, label, file_id)
        video = video.float() / 127.5 - 1
        return video, skeleton, label

    def transform(self):
        if self.transform_mode == "train":
            print("Apply training transform.")
            return video_augmentation.Compose([
                # video_augmentation.CenterCrop(224),
                # video_augmentation.WERAugment('/lustre/wangtao/current_exp/exp/baseline/boundary.npy'),
                video_augmentation.RandomHorizontalFlip_ske(0.5),
                video_augmentation.RandomCrop_ske(224),
                #video_augmentation.resize(224),
                video_augmentation.ToTensor(),
                video_augmentation.TemporalRescale(0.2),
                # video_augmentation.Resize(0.5),
            ])
        else:
            print("Apply testing transform.")
            return video_augmentation.Compose([
                video_augmentation.CenterCrop_ske(224),
                #video_augmentation.Resize(0.5),
                #video_augmentation.resize(224),
                video_augmentation.ToTensor(),
            ])



    @staticmethod
    def collate_fn(batch):
        batch = [item for item in sorted(batch, key=lambda x: len(x[0]), reverse=True)]
        video, skeleton, label, info = list(zip(*batch))
        # padding video
        max_len_v = len(video[0])
        video_length = torch.LongTensor([len(vid) for vid in video])
        padded_video = torch.stack([torch.cat([vid, torch.zeros(vid[0].shape).expand(max_len_v - len(vid), *vid[0].shape)]) for vid in video], axis = 0)

        # padding skeleton
        max_len_s = len(skeleton[0])
        padded_skeleton = torch.stack([torch.cat([ske, torch.zeros(ske[0].shape).expand(max_len_s - len(ske), *ske[0].shape)]) for ske in skeleton], axis = 0)

        #padding label
        max_len_l = max([len(l) for l in label])
        label_length = torch.LongTensor([len(lab) for lab in label])
        padded_label = torch.stack([torch.cat([lab, torch.LongTensor([0] * (max_len_l - len(lab)))])  for lab in label], axis = 0)
        return padded_video, video_length, padded_skeleton, padded_label, label_length, info

    def __len__(self):
        return len(self.inputs_list) - 1
