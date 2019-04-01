import os
import torch
from PIL import Image
import numpy as np
import torch.utils.data as data
from torchvision.datasets.utils import download_url, check_integrity

'''
tiny-imagenet dataset from
<https://github.com/budlbaram/tiny_imagenet/blob/master/tiny_imagenet.py>
'''

IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
]

def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)

def find_classes(class_file):
    with open(class_file) as r:
        classes = [line.strip() for line in r.readlines()]

    classes = sorted(classes)
    class_to_idx = {classes[i]: i for i in range(len(classes))}

    return classes, class_to_idx

def loadPILImage(path):
    trans_img = Image.open(path).convert('RGB')
    return trans_img


def make_dataset(root, base_folder, dirname, class_to_idx):
    images = []
    dir_path = os.path.join(root, base_folder, dirname)

    if dirname == 'train':
        for fname in sorted(os.listdir(dir_path)):
            cls_fpath = os.path.join(dir_path, fname)
            if os.path.isdir(cls_fpath):
                cls_imgs_path = os.path.join(cls_fpath, 'images')
                for imgname in sorted(os.listdir(cls_imgs_path)):
                    if is_image_file(imgname):
                        path = os.path.join(cls_imgs_path, imgname)
                        item = (path, class_to_idx[fname])
                        images.append(item)
    else:
        imgs_path = os.path.join(dir_path, 'images')
        imgs_annotations = os.path.join(dir_path, 'val_annotations.txt')

        with open(imgs_annotations) as r:
            data_info = map(lambda s: s.split('\t'), r.readlines())

        cls_map = {line_data[0]: line_data[1] for line_data in data_info}

        for imgname in sorted(os.listdir(imgs_path)):
            if is_image_file(imgname):
                path = os.path.join(imgs_path, imgname)
                item = (path, class_to_idx[cls_map[imgname]])
                images.append(item)

    return images


class TinyImageNet200(data.Dataset):
    """`tiny-imageNet <http://cs231n.stanford.edu/tiny-imagenet-200.zip>`_ Dataset.
    Args:
        root (string): Root directory of dataset where directory
            ``tiny-imagenet-200`` exists.
        train (bool, optional): If True, creates dataset from training set, otherwise
            creates from test set.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        download (bool, optional): If true, downloads the dataset from the internet and
            puts it in root directory. If dataset is already downloaded, it is not
            downloaded again.
    """

    url = "http://cs231n.stanford.edu/tiny-imagenet-200.zip"
    base_folder = 'tiny-imagenet-200'
    filename = "tiny-imagenet-200.zip"
    md5 = '90528d7ca1a48142e341f4ef8d21d0de'
    filenum = 120203
    dirnum = 405
    def __init__(self, root, train=True, transform=None, target_transform=None, download=False):
        self.root = os.path.expanduser(root)
        self.transform = transform
        self.target_transform = target_transform
        self.train = train  # training set or test set
        self.fpath = os.path.join(root, self.filename)
        self.base_path = os.path.join(root,self.base_folder)
        self.total_size = 0
        self.file_num = 0
        self.dir_num = 0

        if download:
            self.download()

        if not check_integrity(self.fpath, self.md5):
            raise RuntimeError('Dataset not found or corrupted.' +
                               ' You can use download=True to download it')

        _, class_to_idx = find_classes(os.path.join(self.root, self.base_folder, 'wnids.txt'))

        if self.train:
            dirname = 'train'
        else:
            dirname = 'val'

        self.data_info = make_dataset(self.root, self.base_folder, dirname, class_to_idx)

        if len(self.data_info) == 0:
            raise (RuntimeError("Found 0 images in subfolders of: " + root + "\n"
                                                                             "Supported image extensions are: " + ",".join(
                IMG_EXTENSIONS)))

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (img_path, target) where target is index of the target class.
        """

        img_path, target = self.data_info[index][0], self.data_info[index][1]

        img = loadPILImage(img_path)

        if self.transform is not None:
            result_img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return result_img, target

    def __len__(self):
        return len(self.data_info)

    def _count_num_dir_file(self, path):
        for lists in os.listdir(path):
            sub_path = os.path.join(path, lists)
            if os.path.isfile(sub_path):
                self.file_num += 1
            elif os.path.isdir(sub_path):
                self.dir_num += 1
                self._count_num_dir_file(sub_path)

    def _check_integrity(self):
        if not os.path.exists(os.path.join(self.root, self.base_folder)):
            return False
        self._count_num_dir_file(os.path.join(self.root, self.base_folder))
        if self.file_num == self.filenum and self.dir_num == self.dirnum:
            return True

    def download(self):
        import zipfile

        if self._check_integrity():
            if self.train:
                print('Files already downloaded and verified')
            return

        download_url(self.url, self.root, self.filename, self.md5)

        # extract file
        print('extract files to {0}'.format(self.root))
        dataset_zip = zipfile.ZipFile(self.fpath)
        dataset_zip.extractall(self.root)
        dataset_zip.close
        print('extract all ok')




