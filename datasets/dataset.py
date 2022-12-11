import os
from torch.utils.data import Dataset, DataLoader
import cv2
from PIL import Image
import math
import numpy as np
from datasets.Sampler import RandomSampler, BatchSampler
from datasets import transform
from datasets.RandAugment import RandomAugment
from datasets import transform_org
from datasets.RandAugment_org import RandomAugment_o
import shutil
from torchvision import transforms
import random
import torch
import torch.nn.functional as F
import numpy as np
import copy

mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]


class Set(Dataset):
    def __init__(self, imgs, labels, landmarks, is_train, n, m):
        super(Set, self).__init__()
        self.is_train = is_train
        self.imgs = imgs
        self.labels = labels
        self.landmarks = landmarks
        self.resize_size = 224
        self.crop_size = 224
        if self.is_train:
            self.weak_transforms = transform.Compose([
                transform.Resize((self.resize_size, self.resize_size)),
                transform.PadandRandomCrop(border=int(self.resize_size/8), cropsize=(self.crop_size, self.crop_size)),
                transform.RandomHorizontalFlip(p=0.5),
                transform.Normalize(mean, std),
                transform.ToTensor(),
            ])
            self.strong_transforms = transform.Compose([
                transform.Resize((self.resize_size, self.resize_size)),
                transform.PadandRandomCrop(border=int(self.resize_size/8), cropsize=(self.crop_size, self.crop_size)),
                transform.RandomHorizontalFlip(p=0.5),
                RandomAugment(n, m),
                transform.Normalize(mean, std),
                transform.ToTensor(),
                ])
        else:
            self.transforms = transform.Compose([
                transform.Resize((self.crop_size, self.crop_size)),
                transform.Normalize(mean, std),
                transform.ToTensor(),
            ])

    def __getitem__(self, idx):
        img, label, landmark = self.imgs[idx], self.labels[idx], self.landmarks[idx]

        if self.is_train:
            return self.weak_transforms(img, copy.deepcopy(landmark)), self.strong_transforms(img, copy.deepcopy(landmark)), label
        else:
            return self.transforms(img, landmark), label

    def __len__(self):
        return len(self.imgs)


class resample_set(Dataset):
    def __init__(self, imgs, info, landmarks):
        self.imgs = imgs
        self.info = info
        self.landmarks = landmarks
        self.transforms = transform.Compose([
            transform.Resize((224, 224)),
            transform.Normalize(mean, std),
            transform.ToTensor(),
        ])

    def __getitem__(self, idx):
        img, info, landmark = self.imgs[idx], self.info[idx], self.landmarks[idx]

        return self.transforms(img, landmark), info

    def __len__(self):
        return len(self.info)


class pretrain_set(Dataset):
    def __init__(self, imgs, info):
        self.imgs = imgs
        self.info = info
        self.resize_size = 224
        self.crop_size = 224
        self.transforms = transform_org.Compose_o([
                transform_org.Resize((self.resize_size, self.resize_size)),
                transform_org.PadandRandomCrop(border=int(self.resize_size/8), cropsize=(self.crop_size, self.crop_size)),
                transform_org.RandomHorizontalFlip(p=0.5),
                RandomAugment_o(2, 10),
                transform_org.Normalize(mean, std),
                transform_org.ToTensor(),
            ])

    def __getitem__(self, idx):
        img, info = self.imgs[idx], self.info[idx]

        return self.transforms(img), info

    def __len__(self):
        return len(self.info)


def set_random_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class Resample:
    def __init__(self, args):
        set_random_seed()
        self.data = args.data
        self.thr_r = args.thr_r
        self.count = dict()
        f = open(os.path.join('data', args.data, 'train_with_landmark.txt'), 'r')
        train_data = f.readlines()
        if args.n_class == 7:
            self.info_per_class = {
                '0': list(),
                '1': list(),
                '2': list(),
                '3': list(),
                '4': list(),
                '5': list(),
                '6': list(),
            }
        else:
            self.info_per_class = {
                '0': list(),
                '1': list(),
                '2': list(),
                '3': list(),
                '4': list(),
                '5': list(),
                '6': list(),
                '7': list(),
            }
        for line in train_data:
            if not line.endswith('\n'):
                line += '\n'
            self.info_per_class[line.split()[1]].append(line)
        f.close()
        for key in self.info_per_class:
            self.count[key] = len(self.info_per_class[key])

    def resample_image(self, result):
        f = open(os.path.join('data', self.data, 'resample.txt'), 'a+')
        for index in range(len(result)):
            info = result[index][0]
            if not info.endswith('\n'):
                info += '\n'
            image_label = result[index][1]
            random_seed = random.uniform(0.05, 0.2)
            if float(result[index][2]) >= self.thr_r and random.uniform(0, 1) < (min(self.count.values(), key=lambda x:x)/self.count[image_label])**1:
                src = os.path.join('data', self.data, 'image', info.split()[0])
                img = cv2.imread(src)
                f.write(info)
                cv2.imwrite(os.path.join('data', self.data, 'image/resample', info.split()[0]), img)
                # height, width = img.shape[0], img.shape[1]
                # landmark = np.array(info.strip().split()[2:], dtype=np.float32).reshape(4, 2)
                # landmark[:, 0] *= width
                # landmark[:, 1] *= height
                # ground_image_info = random.choice(self.info_per_class[image_label])
                # ground_image = cv2.imread(os.path.join('data', self.data, 'image', ground_image_info.split()[0]))
                # ground_image = cv2.resize(ground_image, (width, height))
                # landmark_g = np.array(ground_image_info.strip().split()[2:], dtype=np.float32).reshape(4, 2)
                # landmark_g[:, 0] *= width
                # landmark_g[:, 1] *= height
                # delta_x = max(1, math.floor(random_seed*width))
                # delta_y = max(1, math.floor(random_seed*height))
                # for i in range(landmark.shape[0]):
                #     ground = copy.deepcopy(ground_image[max(0, math.floor(landmark_g[i][1])-delta_x):min(height, math.floor(landmark_g[i][1])+delta_x), max(0, math.floor(landmark_g[i][0])-delta_y):min(width, math.floor(landmark_g[i][0])+delta_y)])
                #     predict = img[max(0, math.floor(landmark[i][1])-delta_x):min(height, math.floor(landmark[i][1])+delta_x), max(0, math.floor(landmark[i][0])-delta_y):min(width, math.floor(landmark[i][0])+delta_y)]
                #     if predict.shape != ground.shape:
                #         try:
                #             predict = cv2.resize(predict, (ground.shape[1], ground.shape[0]))
                #         except:
                #             a = 1
                #     ground_image[max(0, math.floor(landmark_g[i][1])-delta_x):min(height, math.floor(landmark_g[i][1])+delta_x), max(0, math.floor(landmark_g[i][0])-delta_y):min(width, math.floor(landmark_g[i][0])+delta_y)] = predict
                #
                # new_name =str(ground_image_info.split('.')[0]) + '_' + info.split('.')[0] + '_' + str(random_seed) + '.jpg'
                # try:
                #     f.write(new_name+ground_image_info.split('.jpg')[1])
                # except:
                #     f.write(new_name+ground_image_info.split('.png')[1])
                # cv2.imwrite(os.path.join('data', self.data, 'image/resample', new_name), ground_image)
        f.close()
        self.count[image_label] += 1
        # self.count[image_label] += 2


class dataset(Dataset):
    def __init__(self, args):
        super(dataset, self).__init__()
        set_random_seed()
        self.data = args.data
        self.batch_size = args.batch_size
        self.num_workers = 0
        self.iters = args.iters
        self.mu = args.mu
        self.N = args.aug_n
        self.M = args.aug_m
        root = './data/' + args.data + '/'
        self.train_path = os.path.join(root, 'image')
        self.resample_path = os.path.join(self.train_path, 'resample')
        self.train_file = os.path.join(root, 'train_with_landmark.txt')
        self.val_file = os.path.join(root, 'val_with_landmark.txt')
        self.resample_file = os.path.join(root, 'all_with_landmark.txt')
        # self.resample_file = os.path.join(root, 'train_with_landmark_temp.txt')
        self.extension_file = os.path.join(root, 'resample.txt')
        self.pretrain_train = os.path.join(root, 'train_val.txt')
        self.pretrain_val = os.path.join(root, 'val.txt')
        self.num_labeled = args.label_num
        if self.data != 'Pretrain':
            f = open(self.train_file)
            self.file = f.readlines()
            f.close()
            random.shuffle(self.file)

    def get_train(self):
        dataloader_x, dataloader_u = self.train_data_loader()
        return dataloader_x, dataloader_u

    def get_val(self):
        dataloader_v = self.val_data_loader()
        return dataloader_v

    def get_resample(self):
        dataloader = self.resample_loader()
        return dataloader

    def get_pretrain(self):
        dataloader_t, dataloader_v = self.pretrain_data_loader()
        return dataloader_t, dataloader_v

    def get_pernum(self):
        label = 0
        number_per_emotion = []

        for emotion in os.listdir(self.unlabeled):
            n = 0
            for _ in os.listdir(os.path.join(self.unlabeled, emotion)):
                n += 1
            number_per_emotion.append((label, n))
            label += 1
        return number_per_emotion

    def train_data_loader(self):
        imgs, labels, landmarks = self.load_data('labeled')
        data = Set(imgs, labels, landmarks, is_train=True, n=self.N, m=self.M)
        sampler_data = RandomSampler(data, replacement=True, num_samples=self.iters * self.batch_size)
        batch_data = BatchSampler(sampler_data, batch_size=self.batch_size, drop_last=True)
        dataloader_x = DataLoader(data, batch_sampler=batch_data, num_workers=self.num_workers, pin_memory=True)

        imgs, labels, landmarks = self.load_data('unlabeled')
        data = Set(imgs, labels, landmarks, is_train=True, n=self.N, m=self.M)
        sampler_data = RandomSampler(data, replacement=True, num_samples=self.iters * self.batch_size)
        batch_data = BatchSampler(sampler_data, batch_size=self.batch_size, drop_last=True)
        dataloader_u = DataLoader(data, batch_sampler=batch_data, num_workers=self.num_workers, pin_memory=True)

        return dataloader_x, dataloader_u,

    def val_data_loader(self):
        imgs, labels, landmarks = self.load_data('val')
        data = Set(imgs, labels, landmarks, is_train=False, n=self.N, m=self.M)
        dataloader_v = DataLoader(data, batch_size=self.batch_size, shuffle=False,
                                  drop_last=False, num_workers=self.num_workers, pin_memory=True)

        return dataloader_v

    def resample_loader(self):
        imgs, info, landmarks = self.load_data('resample')
        data = resample_set(imgs, info, landmarks)
        dataloader = DataLoader(data, batch_size=self.batch_size, shuffle=False,
                                drop_last=False, num_workers=self.num_workers, pin_memory=True)

        return dataloader

    def pretrain_data_loader(self):
        imgs, labels, val_imgs, val_labels = self.load_data('pretrain')
        data = pretrain_set(imgs, labels)
        dataloader_t = DataLoader(data, batch_size=self.batch_size, shuffle=True,
                                  drop_last=True, num_workers=self.num_workers, pin_memory=True)
        data = pretrain_set(val_imgs, val_labels)
        dataloader_v = DataLoader(data, batch_size=self.batch_size, shuffle=False,
                                drop_last=False, num_workers=self.num_workers, pin_memory=True)
        return dataloader_t, dataloader_v

    def load_data(self, flag='labeled'):
        imgs, labels, landmarks, info = [], [], [], []

        if flag == 'labeled':
            for line in self.file[:self.num_labeled]:
                data = cv2.cvtColor(cv2.imread(os.path.join(self.train_path, line.split()[0])), cv2.COLOR_BGR2RGB)
                label = int(line.split()[1])
                H, W, _ = data.shape
                landmark = np.array(line.strip().split()[2:], dtype=np.float).reshape((4, 2))
                landmark[:, 0] *= W
                landmark[:, 1] *= H
                imgs.append(data)
                labels.append(label)
                landmarks.append(landmark)
        elif flag == 'unlabeled':
            if self.num_labeled != -1:
                for line in self.file[self.num_labeled:]:
                    data = cv2.cvtColor(cv2.imread(os.path.join(self.train_path, line.split()[0])), cv2.COLOR_BGR2RGB)
                    label = int(line.split()[1])
                    H, W, _ = data.shape
                    landmark = np.array(line.strip().split()[2:], dtype=np.float).reshape((4, 2))
                    landmark[:, 0] *= W
                    landmark[:, 1] *= H
                    imgs.append(data)
                    labels.append(label)
                    landmarks.append(landmark)
            else:
                for line in self.file:
                    data = cv2.cvtColor(cv2.imread(os.path.join(self.train_path, line.split()[0])), cv2.COLOR_BGR2RGB)
                    label = int(line.split()[1])
                    H, W, _ = data.shape
                    landmark = np.array(line.strip().split()[2:], dtype=np.float).reshape((4, 2))
                    landmark[:, 0] *= W
                    landmark[:, 1] *= H
                    imgs.append(data)
                    labels.append(label)
                    landmarks.append(landmark)
        elif flag == 'resample':
            f = open(self.resample_file)
            file = f.readlines()
            f.close()
            for line in file:
                data = cv2.cvtColor(cv2.imread(os.path.join(self.train_path, line.split()[0])), cv2.COLOR_BGR2RGB)
                H, W, _ = data.shape
                landmark = np.array(line.strip().split()[2:], dtype=np.float).reshape((4, 2))
                landmark[:, 0] *= W
                landmark[:, 1] *= H
                imgs.append(data)
                info.append(line)
                landmarks.append(landmark)

            return imgs, info, landmarks
        elif flag == 'pretrain':
            f = open(self.pretrain_train)
            file = f.readlines()
            f.close()
            for line in file:
                try:
                    data = cv2.cvtColor(cv2.imread(os.path.join(self.train_path, line.split()[0])), cv2.COLOR_BGR2RGB)
                except:
                    print(line)
                    continue
                imgs.append(data)
                labels.append(int(line.split()[1]))
            val_imgs, val_labels = [], []
            f = open(self.pretrain_val)
            file = f.readlines()
            f.close()
            for line in file:
                try:
                    data = cv2.cvtColor(cv2.imread(os.path.join(self.train_path, line.split()[0])), cv2.COLOR_BGR2RGB)
                except:
                    print(line)
                    continue
                val_imgs.append(data)
                val_labels.append(int(line.split()[1]))
            return imgs, labels, val_imgs, val_labels
        else:
            f = open(self.val_file)
            file = f.readlines()
            f.close()
            for line in file:
                data = cv2.cvtColor(cv2.imread(os.path.join(self.train_path, line.split()[0])), cv2.COLOR_BGR2RGB)
                label = int(line.split()[1])
                H, W, _ = data.shape
                landmark = np.array(line.strip().split()[2:], dtype=np.float).reshape((4, 2))
                landmark[:, 0] *= W
                landmark[:, 1] *= H
                imgs.append(data)
                labels.append(label)
                landmarks.append(landmark)

        if len(os.listdir(self.resample_path)) > 0 and flag != 'val':
            f = open(self.extension_file, 'r')
            file = f.readlines()
            f.close()
            for line in file:
                data = cv2.cvtColor(cv2.imread(os.path.join(self.resample_path, line.split()[0])), cv2.COLOR_BGR2RGB)
                label = int(line.split()[1])
                H, W, _ = data.shape
                landmark = np.array(line.strip().split()[2:], dtype=np.float).reshape((4, 2))
                landmark[:, 0] *= W
                landmark[:, 1] *= H
                imgs.append(data)
                labels.append(label)
                landmarks.append(landmark)

        return imgs, labels, landmarks


def main():
    source = 'CK+'
    target = 'RAF-DB'
    dataloader_s, dataloader_t, dataloader_v = dataset(source, target, iters=10)
    dataloader_s = iter(dataloader_s)
    dataloader_t = iter(dataloader_t)
    for i in range(10):
        img_weak_x, img_strong_x, label_x = next(dataloader_s)
    a = 1


if __name__ == '__main__':
    main()
