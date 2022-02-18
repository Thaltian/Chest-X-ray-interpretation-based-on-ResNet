import torch
import os, glob
import random, csv

from    torch.utils.data import Dataset, DataLoader

from    torchvision import transforms
from    PIL import Image


class xray(Dataset):

    def __init__(self, root, resize, mode):
        super(xray, self).__init__()

        self.root = root
        self.resize = resize

        self.name2label = {} # "sq...":0
        for name in sorted(os.listdir(os.path.join(root))):
            if not os.path.isdir(os.path.join(root, name)):
                continue

            for name2 in sorted(os.listdir(os.path.join(root,name))):
                if not os.path.isdir(os.path.join(root, name)):
                    continue
            self.name2label[name] = len(self.name2label.keys())

        # print(self.name2label)

        # image, label
        if mode=='train':
            self.images, self.labels = self.load_csv('images_train.csv')
        elif mode=='val':
            self.images, self.labels = self.load_csv('images_val.csv')
        else:
            self.images, self.labels = self.load_csv('images_test.csv')
        '''
        if mode=='train': # 60%
            pass
        elif mode=='val': # 20% = 60%->80%
            self.images = self.images[:int(0.25 * len(self.images))]
            self.labels = self.labels[:int(0.25 * len(self.labels))]
        else: # 20% = 80%->100%
            self.images = self.images[int(0.25*len(self.images)):]
            self.labels = self.labels[int(0.25*len(self.labels)):]

        '''



    def load_csv(self, filename):
        parent_path = os.path.dirname(self.root)
        if not os.path.exists(os.path.join(parent_path, 'csv',filename)):
            images = []
            for name in self.name2label.keys():
                # 'face_test\\fake_face\\0011\\0011_01_01_03_105.jpg'
                #for name2 in sorted(os.listdir(os.path.join(self.root,name))):
                #images += glob.glob(os.path.join(self.root, name, name2,'*.jpg'))
                images += glob.glob(os.path.join(self.root, name, '*.jpg'))

            # number, 'face_test\\fake_face\\0011\\0011_01_01_03_105.jpg'
            print(len(images), images)

            random.shuffle(images)
            with open(os.path.join(parent_path, 'csv', filename), mode='w', newline='') as f:
                writer = csv.writer(f)
                for img in images: #  'face_test\\fake_face\\0011\\0011_01_01_03_105.jpg'
                    name = img.split(os.sep)[-2]
                    label = self.name2label[name]
                    #  'face_test\\fake_face\\0011\\0011_01_01_03_105.jpg' 0
                    writer.writerow([img, label])
                print('writen into csv file:', filename)

        # read from csv file
        images, labels = [], []
        with open(os.path.join(parent_path, 'csv', filename)) as f:
            reader = csv.reader(f)
            for row in reader:
                 # 'face_test\\fake_face\\0011\\0011_01_01_03_105.jpg' 0
                img, label = row
                label = int(label)

                images.append(img)
                labels.append(label)

        assert len(images) == len(labels)

        return images, labels



    def __len__(self):

        return len(self.images)


    def denormalize(self, x_hat):

        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]

        # x_hat = (x-mean)/std
        # x = x_hat*std = mean
        # x: [c, h, w]
        # mean: [3] => [3, 1, 1]
        mean = torch.tensor(mean).unsqueeze(1).unsqueeze(1)
        std = torch.tensor(std).unsqueeze(1).unsqueeze(1)
        # print(mean.shape, std.shape)
        x = x_hat * std + mean

        return x


    def __getitem__(self, idx):
        # idx~[0~len(images)]
        # self.images, self.labels
        # img: ' 'face_test\\fake_face\\0011\\0011_01_01_03_105.jpg'
        # label: 0
        img, label = self.images[idx], self.labels[idx]

        tf = transforms.Compose([
            lambda x:Image.open(x).convert('RGB'), # string path= > image data
            transforms.Resize((int(self.resize), int(self.resize))),
            # transforms.RandomRotation(15),
            # transforms.CenterCrop(self.resize),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

        img = tf(img)
        label = torch.tensor(label)


        return img, label





