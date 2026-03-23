# YOLOv3 🚀 by Ultralytics, GPL-3.0 license
"""
Dataloaders and dataset utils
"""

import glob
import hashlib
import json
import os
import random
import shutil
import time
from itertools import repeat
from multiprocessing.pool import Pool, ThreadPool
from pathlib import Path
from threading import Thread
from zipfile import ZipFile

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import yaml
from PIL import ExifTags, Image, ImageOps
from torch.utils.data import DataLoader, Dataset, dataloader, distributed
from tqdm import tqdm

from utils.augmentations import Albumentations, augment_hsv, copy_paste, letterbox, mixup, random_perspective
from utils.general import (LOGGER, check_dataset, check_requirements, check_yaml, clean_str, segments2boxes, xyn2xy,
                           xywh2xyxy, xywhn2xyxy, xyxy2xywhn)
from utils.torch_utils import torch_distributed_zero_first

# Parameters
HELP_URL = 'https://github.com/ultralytics/yolov3/wiki/Train-Custom-Data'
IMG_FORMATS = ['bmp', 'jpg', 'jpeg', 'png', 'tif', 'tiff', 'dng', 'webp', 'mpo']  # acceptable image suffixes
VID_FORMATS = ['mov', 'avi', 'mp4', 'mpg', 'mpeg', 'm4v', 'wmv', 'mkv']  # acceptable video suffixes
WORLD_SIZE = int(os.getenv('WORLD_SIZE', 1))  # DPP
NUM_THREADS = min(8, os.cpu_count())  # number of multiprocessing threads

# Get orientation exif tag
for orientation in ExifTags.TAGS.keys():
    if ExifTags.TAGS[orientation] == 'Orientation':
        break

def get_hash(paths):
    # Returns a single hash value of a list of paths (files or dirs)
    size = sum(os.path.getsize(p) for p in paths if os.path.exists(p))  # 计算存在路径的总大小
    h = hashlib.md5(str(size).encode())  # 基于大小创建 MD5 哈希
    h.update(''.join(paths).encode())  # 追加路径字符串的哈希
    return h.hexdigest()  # 返回哈希值

def exif_size(img):
    # Returns exif-corrected PIL size
    s = img.size  # 获取图片的原始大小 (宽度, 高度)
    try:
        # 从图片的 EXIF 数据中获取方向信息
        rotation = dict(img._getexif().items())[orientation]
        if rotation == 6:  # 如果方向为 6，表示需要旋转 270 度
            s = (s[1], s[0])  # 交换宽度和高度
        elif rotation == 8:  # 如果方向为 8，表示需要旋转 90 度
            s = (s[1], s[0])  # 交换宽度和高度
    except:
        pass  # 如果获取 EXIF 数据失败，则保持原始大小

    return s  # 返回调整后的大小



def exif_transpose(image):
    """
    Transpose a PIL image accordingly if it has an EXIF Orientation tag.
    Inplace version of https://github.com/python-pillow/Pillow/blob/master/src/PIL/ImageOps.py exif_transpose()

    :param image: The image to transpose.
    :return: An image.
    """
    exif = image.getexif()  # 获取图片的 EXIF 数据
    orientation = exif.get(0x0112, 1)  # 获取方向信息，默认为 1
    if orientation > 1:  # 如果方向信息大于 1
        # 根据方向信息选择相应的变换方法
        method = {2: Image.FLIP_LEFT_RIGHT,    # 水平翻转
                  3: Image.ROTATE_180,       # 旋转 180 度
                  4: Image.FLIP_TOP_BOTTOM,   # 垂直翻转
                  5: Image.TRANSPOSE,         # 转置
                  6: Image.ROTATE_270,        # 旋转 270 度
                  7: Image.TRANSVERSE,        # 反转转置
                  8: Image.ROTATE_90,         # 旋转 90 度
                  }.get(orientation)  # 根据方向获取相应的方法
        if method is not None:  # 如果找到了变换方法
            image = image.transpose(method)  # 对图片进行变换
            del exif[0x0112]  # 删除方向信息
            image.info["exif"] = exif.tobytes()  # 更新图片的 EXIF 信息
    return image  # 返回变换后的图片


def create_dataloader(path, imgsz, batch_size, stride, single_cls=False, hyp=None, augment=False, cache=False, pad=0.0,
                      rect=False, rank=-1, workers=8, image_weights=False, quad=False, prefix='', shuffle=False):
    # 检查矩形模式与打乱数据的兼容性
    if rect and shuffle:
        LOGGER.warning('WARNING: --rect is incompatible with DataLoader shuffle, setting shuffle=False')
        shuffle = False

    with torch_distributed_zero_first(rank):  # 在分布式训练中，仅初始化一次数据集 *.cache
        dataset = LoadImagesAndLabels(path, imgsz, batch_size,
                                       augment=augment,  # 数据增强
                                       hyp=hyp,  # 超参数
                                       rect=rect,  # 矩形批次
                                       cache_images=cache,
                                       single_cls=single_cls,
                                       stride=int(stride),
                                       pad=pad,
                                       image_weights=image_weights,
                                       prefix=prefix)

    batch_size = min(batch_size, len(dataset))  # 确保批次大小不超过数据集大小
    nw = min([os.cpu_count() // WORLD_SIZE, batch_size if batch_size > 1 else 0, workers])  # 计算工作线程数量
    sampler = None if rank == -1 else distributed.DistributedSampler(dataset, shuffle=shuffle)  # 创建采样器

    # 选择合适的 DataLoader
    loader = DataLoader if image_weights else InfiniteDataLoader  # 仅 DataLoader 支持属性更新

    return loader(dataset,
                  batch_size=batch_size,
                  shuffle=shuffle and sampler is None,  # 在没有采样器时打乱数据
                  num_workers=nw,  # 工作线程数量
                  sampler=sampler,  # 数据采样器
                  pin_memory=True,  # 固定内存
                  collate_fn=LoadImagesAndLabels.collate_fn4 if quad else LoadImagesAndLabels.collate_fn), dataset



class InfiniteDataLoader(dataloader.DataLoader):
    """ Dataloader that reuses workers

    Uses same syntax as vanilla DataLoader
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)  # 初始化父类 DataLoader
        # 设置一个重复采样器，确保可以无限次使用数据
        object.__setattr__(self, 'batch_sampler', _RepeatSampler(self.batch_sampler))
        self.iterator = super().__iter__()  # 获取迭代器

    def __len__(self):
        return len(self.batch_sampler.sampler)  # 返回样本总数

    def __iter__(self):
        # 无限迭代，重新生成迭代器
        for i in range(len(self)):
            yield next(self.iterator)  # 返回下一个样本



class _RepeatSampler:
    """ Sampler that repeats forever

    Args:
        sampler (Sampler): 要重复的采样器
    """

    def __init__(self, sampler):
        self.sampler = sampler  # 保存传入的采样器

    def __iter__(self):
        # 无限迭代，重复返回采样器中的元素
        while True:
            yield from iter(self.sampler)  # 逐个生成采样器中的元素


class LoadImages:
    # 图像/视频数据加载器，例如 `python detect.py --source image.jpg/vid.mp4`

    def __init__(self, path, img_size=640, stride=32, auto=True):
        # 解析并准备文件路径
        p = str(Path(path).resolve())  # 获取平台无关的绝对路径
        if '*' in p:
            files = sorted(glob.glob(p, recursive=True))  # 使用 glob 获取匹配的文件
        elif os.path.isdir(p):
            files = sorted(glob.glob(os.path.join(p, '*.*')))  # 获取目录下的所有文件
        elif os.path.isfile(p):
            files = [p]  # 单个文件
        else:
            raise Exception(f'ERROR: {p} does not exist')  # 文件或目录不存在

        # 根据文件扩展名分类图像和视频
        images = [x for x in files if x.split('.')[-1].lower() in IMG_FORMATS]
        videos = [x for x in files if x.split('.')[-1].lower() in VID_FORMATS]
        ni, nv = len(images), len(videos)  # 统计图像和视频的数量

        self.img_size = img_size  # 设置图像大小
        self.stride = stride  # 设置步长
        self.files = images + videos  # 合并图像和视频文件列表
        self.nf = ni + nv  # 文件总数
        self.video_flag = [False] * ni + [True] * nv  # 视频标志列表
        self.mode = 'image'  # 当前模式，默认为图像
        self.auto = auto  # 是否自动调整
        if any(videos):
            self.new_video(videos[0])  # 初始化第一个视频
        else:
            self.cap = None  # 如果没有视频，设置为 None

        # 检查是否找到任何有效文件
        assert self.nf > 0, f'No images or videos found in {p}. ' \
                            f'Supported formats are:\nimages: {IMG_FORMATS}\nvideos: {VID_FORMATS}'

    def __iter__(self):
        self.count = 0  # 重置计数器
        return self

    def __next__(self):
        if self.count == self.nf:
            raise StopIteration  # 达到文件末尾，停止迭代
        path = self.files[self.count]  # 获取当前文件路径

        if self.video_flag[self.count]:
            # 读取视频帧
            self.mode = 'video'  # 设置模式为视频
            ret_val, img0 = self.cap.read()  # 读取视频帧
            if not ret_val:  # 如果未能读取帧
                self.count += 1  # 移动到下一个文件
                self.cap.release()  # 释放当前视频捕获对象
                if self.count == self.nf:  # 如果是最后一个视频
                    raise StopIteration
                else:
                    path = self.files[self.count]  # 获取下一个文件路径
                    self.new_video(path)  # 初始化新视频
                    ret_val, img0 = self.cap.read()  # 读取新视频帧

            self.frame += 1  # 帧计数增加
            s = f'video {self.count + 1}/{self.nf} ({self.frame}/{self.frames}) {path}: '

        else:
            # 读取图像
            self.count += 1  # 移动到下一个文件
            img0 = cv2.imread(path)  # 使用 OpenCV 读取图像（BGR格式）
            assert img0 is not None, f'Image Not Found {path}'  # 检查图像是否有效
            s = f'image {self.count}/{self.nf} {path}: '

        # 填充调整大小
        img = letterbox(img0, self.img_size, stride=self.stride, auto=self.auto)[0]

        # 转换格式
        img = img.transpose((2, 0, 1))[::-1]  # 从 HWC 转换为 CHW，并从 BGR 转为 RGB
        img = np.ascontiguousarray(img)  # 确保数组是连续的

        return path, img, img0, self.cap, s  # 返回路径、处理后的图像、原始图像、视频捕获对象和状态信息

    def new_video(self, path):
        self.frame = 0  # 重置帧计数
        self.cap = cv2.VideoCapture(path)  # 初始化视频捕获
        self.frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))  # 获取视频总帧数

    def __len__(self):
        return self.nf  # 返回文件总数


class LoadWebcam:  # 用于推理
    # 本地摄像头数据加载器，例如 `python detect.py --source 0`
    def __init__(self, pipe='0', img_size=640, stride=32):
        self.img_size = img_size  # 设置图像大小
        self.stride = stride  # 设置步长
        self.pipe = eval(pipe) if pipe.isnumeric() else pipe  # 解析管道参数
        self.cap = cv2.VideoCapture(self.pipe)  # 创建视频捕获对象
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 3)  # 设置缓冲区大小

    def __iter__(self):
        self.count = -1  # 初始化计数器
        return self

    def __next__(self):
        self.count += 1  # 增加计数器
        if cv2.waitKey(1) == ord('q'):  # 如果按下 'q' 键，则退出
            self.cap.release()  # 释放视频捕获对象
            cv2.destroyAllWindows()  # 关闭所有窗口
            raise StopIteration  # 停止迭代

        # 读取帧
        ret_val, img0 = self.cap.read()  # 从摄像头读取图像
        img0 = cv2.flip(img0, 1)  # 水平翻转图像

        # 打印状态
        assert ret_val, f'Camera Error {self.pipe}'  # 检查摄像头是否正常
        img_path = 'webcam.jpg'  # 设置图像路径
        s = f'webcam {self.count}: '  # 状态信息

        # 填充调整大小
        img = letterbox(img0, self.img_size, stride=self.stride)[0]

        # 转换格式
        img = img.transpose((2, 0, 1))[::-1]  # 从 HWC 转换为 CHW，并从 BGR 转为 RGB
        img = np.ascontiguousarray(img)  # 确保数组是连续的

        return img_path, img, img0, None, s  # 返回图像路径、处理后的图像、原始图像、None 和状态信息

    def __len__(self):
        return 0  # 返回 0，表示无限循环


class LoadStreams:
    # 流加载器，例如 `python detect.py --source 'rtsp://example.com/media.mp4'`  # 支持 RTSP、RTMP、HTTP 流

    def __init__(self, sources='streams.txt', img_size=640, stride=32, auto=True):
        self.mode = 'stream'  # 设置模式为流
        self.img_size = img_size  # 设置图像大小
        self.stride = stride  # 设置步长

        # 处理输入源，读取文件或直接使用源字符串
        if os.path.isfile(sources):
            with open(sources) as f:
                sources = [x.strip() for x in f.read().strip().splitlines() if len(x.strip())]  # 从文件读取源
        else:
            sources = [sources]  # 将源设置为单个字符串

        n = len(sources)  # 源的数量
        # 初始化图像、FPS、帧数和线程的列表
        self.imgs, self.fps, self.frames, self.threads = [None] * n, [0] * n, [0] * n, [None] * n
        self.sources = [clean_str(x) for x in sources]  # 清理源名称以便后续使用
        self.auto = auto  # 自动调整标志

        for i, s in enumerate(sources):  # 遍历每个源
            # 启动线程以从视频流读取帧
            st = f'{i + 1}/{n}: {s}... '
            if 'youtube.com/' in s or 'youtu.be/' in s:  # 如果源是 YouTube 视频
                check_requirements(('pafy', 'youtube_dl'))  # 检查所需库
                import pafy
                s = pafy.new(s).getbest(preftype="mp4").url  # 获取最佳 YouTube URL
            s = eval(s) if s.isnumeric() else s  # 处理本地摄像头源
            cap = cv2.VideoCapture(s)  # 创建视频捕获对象
            assert cap.isOpened(), f'{st}Failed to open {s}'  # 确保成功打开流
            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))  # 获取流的宽度
            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))  # 获取流的高度
            self.fps[i] = max(cap.get(cv2.CAP_PROP_FPS) % 100, 0) or 30.0  # 获取 FPS，若无法获取则默认为 30 FPS
            self.frames[i] = max(int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), 0) or float('inf')  # 获取帧数，默认为无限流

            _, self.imgs[i] = cap.read()  # 确保读取第一帧
            self.threads[i] = Thread(target=self.update, args=([i, cap, s]), daemon=True)  # 创建线程以更新帧
            LOGGER.info(f"{st} Success ({self.frames[i]} frames {w}x{h} at {self.fps[i]:.2f} FPS)")  # 日志输出
            self.threads[i].start()  # 启动线程
        LOGGER.info('')  # 输出换行

        # 检查图像形状是否一致
        s = np.stack([letterbox(x, self.img_size, stride=self.stride, auto=self.auto)[0].shape for x in self.imgs])
        self.rect = np.unique(s, axis=0).shape[0] == 1  # 如果所有形状相同，则进行矩形推理
        if not self.rect:
            LOGGER.warning('WARNING: Stream shapes differ. For optimal performance supply similarly-shaped streams.')

    def update(self, i, cap, stream):
        # 在守护线程中读取流 `i` 的帧
        n, f, read = 0, self.frames[i], 1  # 帧计数、帧数组、每 'read' 帧推理一次
        while cap.isOpened() and n < f:  # 循环直到流关闭或读取完帧
            n += 1
            cap.grab()  # 抓取下一帧
            if n % read == 0:  # 每 'read' 帧进行一次读取
                success, im = cap.retrieve()  # 尝试获取帧
                if success:
                    self.imgs[i] = im  # 更新图像
                else:
                    LOGGER.warning('WARNING: Video stream unresponsive, please check your IP camera connection.')
                    self.imgs[i] *= 0  # 如果失败，将图像置为 0
                    cap.open(stream)  # 重新打开流
            time.sleep(1 / self.fps[i])  # 根据 FPS 等待

    def __iter__(self):
        self.count = -1  # 初始化计数器
        return self

    def __next__(self):
        self.count += 1
        if not all(x.is_alive() for x in self.threads) or cv2.waitKey(1) == ord('q'):  # 检查线程是否存活，或按 'q' 键退出
            cv2.destroyAllWindows()  # 关闭所有窗口
            raise StopIteration  # 停止迭代

        # 进行填充调整
        img0 = self.imgs.copy()  # 复制当前图像
        img = [letterbox(x, self.img_size, stride=self.stride, auto=self.rect and self.auto)[0] for x in img0]

        # 堆叠图像
        img = np.stack(img, 0)

        # 转换格式
        img = img[..., ::-1].transpose((0, 3, 1, 2))  # BGR 转为 RGB，从 BHWC 转为 BCHW
        img = np.ascontiguousarray(img)  # 确保数组是连续的

        return self.sources, img, img0, None, ''  # 返回源、处理后的图像、原始图像、None 和空字符串

    def __len__(self):
        return len(self.sources)  # 返回源的数量



def img2label_paths(img_paths):
    # 根据图像路径定义标签路径
    sa, sb = os.sep + 'images' + os.sep, os.sep + 'labels' + os.sep  # 定义 /images/ 和 /labels/ 的子字符串
    return [sb.join(x.rsplit(sa, 1)).rsplit('.', 1)[0] + '.txt' for x in img_paths]  # 将图像路径转换为标签路径


class LoadImagesAndLabels(Dataset):
    # 训练加载器/验证加载器，用于加载图像和标签
    cache_version = 0.6  # 数据集标签缓存版本

    def __init__(self, path, img_size=640, batch_size=16, augment=False, hyp=None, rect=False, image_weights=False,
                 cache_images=False, single_cls=False, stride=32, pad=0.0, prefix=''):
        # 初始化参数
        self.img_size = img_size  # 图像大小
        self.augment = augment  # 是否进行数据增强
        self.hyp = hyp  # 超参数
        self.image_weights = image_weights  # 是否使用图像权重
        self.rect = False if image_weights else rect  # 是否进行矩形训练
        self.mosaic = self.augment and not self.rect  # 训练时是否使用马赛克增强
        self.mosaic_border = [-img_size // 2, -img_size // 2]  # 马赛克边界
        self.stride = stride  # 步幅
        self.path = path  # 数据路径
        self.albumentations = Albumentations() if augment else None  # 如果启用增强，则初始化Albumentations

        try:
            f = []  # 图像文件列表
            for p in path if isinstance(path, list) else [path]:
                p = Path(p)  # 使路径平台无关
                if p.is_dir():  # 如果是目录
                    f += glob.glob(str(p / '**' / '*.*'), recursive=True)  # 递归获取所有图像文件
                elif p.is_file():  # 如果是文件
                    with open(p) as t:
                        t = t.read().strip().splitlines()  # 读取文件内容
                        parent = str(p.parent) + os.sep
                        f += [x.replace('./', parent) if x.startswith('./') else x for x in t]  # 替换路径
                else:
                    raise Exception(f'{prefix}{p} does not exist')  # 抛出异常
            self.img_files = sorted(x.replace('/', os.sep) for x in f if x.split('.')[-1].lower() in IMG_FORMATS)
            # 筛选有效的图像文件
            assert self.img_files, f'{prefix}No images found'  # 确保找到图像文件
        except Exception as e:
            raise Exception(f'{prefix}Error loading data from {path}: {e}\nSee {HELP_URL}')  # 错误处理

        # 检查缓存
        self.label_files = img2label_paths(self.img_files)  # 获取标签文件路径
        cache_path = (p if p.is_file() else Path(self.label_files[0]).parent).with_suffix('.cache')
        try:
            cache, exists = np.load(cache_path, allow_pickle=True).item(), True  # 加载缓存
            assert cache['version'] == self.cache_version  # 确保版本一致
            assert cache['hash'] == get_hash(self.label_files + self.img_files)  # 确保哈希一致
        except:
            cache, exists = self.cache_labels(cache_path, prefix), False  # 如果缓存无效，则重新缓存

        # 显示缓存信息
        nf, nm, ne, nc, n = cache.pop('results')  # 获取缓存统计信息
        if exists:
            d = f"Scanning '{cache_path}' images and labels... {nf} found, {nm} missing, {ne} empty, {nc} corrupted"
            tqdm(None, desc=prefix + d, total=n, initial=n)  # 显示缓存结果
            if cache['msgs']:
                LOGGER.info('\n'.join(cache['msgs']))  # 显示警告信息
        assert nf > 0 or not augment, f'{prefix}No labels in {cache_path}. Can not train without labels. See {HELP_URL}'  # 确保有标签

        # 读取缓存
        [cache.pop(k) for k in ('hash', 'version', 'msgs')]  # 移除不必要的项
        labels, shapes, self.segments = zip(*cache.values())  # 解压标签和形状
        self.labels = list(labels)  # 标签列表
        self.shapes = np.array(shapes, dtype=np.float64)  # 形状数组
        self.img_files = list(cache.keys())  # 更新图像文件
        self.label_files = img2label_paths(cache.keys())  # 更新标签文件
        n = len(shapes)  # 图像数量
        bi = np.floor(np.arange(n) / batch_size).astype(int)  # 计算批次索引
        nb = bi[-1] + 1  # 批次数
        self.batch = bi  # 记录批次索引
        self.n = n  # 图像总数
        self.indices = range(n)  # 索引范围

        # 更新标签
        include_class = []  # 过滤标签以仅包含这些类（可选）
        include_class_array = np.array(include_class).reshape(1, -1)  # 转换为数组
        for i, (label, segment) in enumerate(zip(self.labels, self.segments)):
            if include_class:  # 如果指定了类
                j = (label[:, 0:1] == include_class_array).any(1)  # 过滤类
                self.labels[i] = label[j]
                if segment:
                    self.segments[i] = segment[j]
            if single_cls:  # 单类训练，将所有类合并为0
                self.labels[i][:, 0] = 0
                if segment:
                    self.segments[i][:, 0] = 0

        # 矩形训练
        if self.rect:
            # 按长宽比排序
            s = self.shapes  # 形状
            ar = s[:, 1] / s[:, 0]  # 计算长宽比
            irect = ar.argsort()  # 排序索引
            self.img_files = [self.img_files[i] for i in irect]  # 更新图像文件
            self.label_files = [self.label_files[i] for i in irect]  # 更新标签文件
            self.labels = [self.labels[i] for i in irect]  # 更新标签
            self.shapes = s[irect]  # 更新形状
            ar = ar[irect]  # 更新长宽比

            # 设置训练图像的形状
            shapes = [[1, 1]] * nb
            for i in range(nb):
                ari = ar[bi == i]  # 获取当前批次的长宽比
                mini, maxi = ari.min(), ari.max()  # 最小和最大长宽比
                if maxi < 1:
                    shapes[i] = [maxi, 1]  # 设置形状
                elif mini > 1:
                    shapes[i] = [1, 1 / mini]  # 设置形状

            self.batch_shapes = np.ceil(np.array(shapes) * img_size / stride + pad).astype(int) * stride  # 计算批次形状

        # 将图像缓存到内存以加快训练速度（警告：大型数据集可能超过系统内存）
        self.imgs, self.img_npy = [None] * n, [None] * n
        if cache_images:
            if cache_images == 'disk':  # 如果缓存到磁盘
                self.im_cache_dir = Path(Path(self.img_files[0]).parent.as_posix() + '_npy')  # 缓存目录
                self.img_npy = [self.im_cache_dir / Path(f).with_suffix('.npy').name for f in self.img_files]  # 缓存文件路径
                self.im_cache_dir.mkdir(parents=True, exist_ok=True)  # 创建缓存目录
            gb = 0  # 缓存图像的大小（GB）
            self.img_hw0, self.img_hw = [None] * n, [None] * n  # 原始和调整后的图像尺寸
            results = ThreadPool(NUM_THREADS).imap(lambda x: load_image(*x), zip(repeat(self), range(n)))  # 多线程加载图像
            pbar = tqdm(enumerate(results), total=n)  # 显示进度条
            for i, x in pbar:
                if cache_images == 'disk':
                    if not self.img_npy[i].exists():  # 如果缓存文件不存在
                        np.save(self.img_npy[i].as_posix(), x[0])  # 保存到磁盘
                    gb += self.img_npy[i].stat().st_size  # 更新缓存大小
                else:
                    self.imgs[i], self.img_hw0[i], self.img_hw[i] = x  # 加载图像及其尺寸
                    gb += self.imgs[i].nbytes  # 更新缓存大小
                pbar.desc = f'{prefix}Caching images ({gb / 1E9:.1f}GB {cache_images})'  # 更新进度描述
            pbar.close()  # 关闭进度条

    def cache_labels(self, path=Path('./labels.cache'), prefix=''):
        # 缓存数据集标签，检查图像并读取形状
        x = {}  # 初始化字典
        nm, nf, ne, nc, msgs = 0, 0, 0, 0, []  # 计数：缺失、找到、空、损坏的标签及消息
        desc = f"{prefix}Scanning '{path.parent / path.stem}' images and labels..."  # 描述信息
        with Pool(NUM_THREADS) as pool:  # 创建多线程池
            pbar = tqdm(pool.imap(verify_image_label, zip(self.img_files, self.label_files, repeat(prefix))),
                        desc=desc, total=len(self.img_files))  # 显示进度条
            for im_file, l, shape, segments, nm_f, nf_f, ne_f, nc_f, msg in pbar:
                # 更新计数
                nm += nm_f
                nf += nf_f
                ne += ne_f
                nc += nc_f
                if im_file:  # 如果找到图像文件
                    x[im_file] = [l, shape, segments]  # 保存文件信息
                if msg:  # 如果有消息
                    msgs.append(msg)  # 记录消息
                pbar.desc = f"{desc}{nf} found, {nm} missing, {ne} empty, {nc} corrupted"  # 更新进度描述

        pbar.close()  # 关闭进度条
        if msgs:  # 如果有消息
            LOGGER.info('\n'.join(msgs))  # 记录消息
        if nf == 0:  # 如果没有找到任何标签
            LOGGER.warning(f'{prefix}WARNING: No labels found in {path}. See {HELP_URL}')  # 发出警告
        x['hash'] = get_hash(self.label_files + self.img_files)  # 生成哈希值
        x['results'] = nf, nm, ne, nc, len(self.img_files)  # 缓存结果信息
        x['msgs'] = msgs  # 记录警告消息
        x['version'] = self.cache_version  # 缓存版本
        try:
            np.save(path, x)  # 保存缓存以备下次使用
            path.with_suffix('.cache.npy').rename(path)  # 移除 .npy 后缀
            LOGGER.info(f'{prefix}New cache created: {path}')  # 记录新缓存创建的信息
        except Exception as e:
            LOGGER.warning(f'{prefix}WARNING: Cache directory {path.parent} is not writeable: {e}')  # 缓存目录不可写
        return x  # 返回缓存字典

    def __len__(self):
        return len(self.img_files)  # 返回图像文件的数量

    def __getitem__(self, index):
        index = self.indices[index]  # 获取线性、随机或基于图像权重的索引

        hyp = self.hyp  # 超参数
        mosaic = self.mosaic and random.random() < hyp['mosaic']  # 根据概率决定是否使用马赛克增强
        if mosaic:
            # 加载马赛克图像
            img, labels = load_mosaic(self, index)
            shapes = None

            # MixUp增强
            if random.random() < hyp['mixup']:
                img, labels = mixup(img, labels, *load_mosaic(self, random.randint(0, self.n - 1)))

        else:
            # 加载单幅图像
            img, (h0, w0), (h, w) = load_image(self, index)

            # 进行信箱填充
            shape = self.batch_shapes[self.batch[index]] if self.rect else self.img_size  # 最终填充后的形状
            img, ratio, pad = letterbox(img, shape, auto=False, scaleup=self.augment)
            shapes = (h0, w0), ((h / h0, w / w0), pad)  # 用于COCO mAP的重标定

            labels = self.labels[index].copy()  # 复制标签
            if labels.size:  # 如果有标签，进行归一化xywh转为像素xyxy格式
                labels[:, 1:] = xywhn2xyxy(labels[:, 1:], ratio[0] * w, ratio[1] * h, padw=pad[0], padh=pad[1])

            if self.augment:
                # 随机透视变换增强
                img, labels = random_perspective(img, labels,
                                                 degrees=hyp['degrees'],
                                                 translate=hyp['translate'],
                                                 scale=hyp['scale'],
                                                 shear=hyp['shear'],
                                                 perspective=hyp['perspective'])

        nl = len(labels)  # 标签数量
        if nl:
            labels[:, 1:5] = xyxy2xywhn(labels[:, 1:5], w=img.shape[1], h=img.shape[0], clip=True, eps=1E-3)  # 转换标签格式

        if self.augment:
            # 使用Albumentations进行数据增强
            img, labels = self.albumentations(img, labels)
            nl = len(labels)  # 更新标签数量

            # HSV色彩空间增强
            augment_hsv(img, hgain=hyp['hsv_h'], sgain=hyp['hsv_s'], vgain=hyp['hsv_v'])

            # 垂直翻转
            if random.random() < hyp['flipud']:
                img = np.flipud(img)
                if nl:
                    labels[:, 2] = 1 - labels[:, 2]  # 更新标签

            # 水平翻转
            if random.random() < hyp['fliplr']:
                img = np.fliplr(img)
                if nl:
                    labels[:, 1] = 1 - labels[:, 1]  # 更新标签

            # Cutouts（可以选择性开启）
            # labels = cutout(img, labels, p=0.5)

        labels_out = torch.zeros((nl, 6))  # 初始化输出标签
        if nl:
            labels_out[:, 1:] = torch.from_numpy(labels)  # 将标签转为torch张量

        # 转换图像格式
        img = img.transpose((2, 0, 1))[::-1]  # HWC转CHW，BGR转RGB
        img = np.ascontiguousarray(img)  # 确保数组是连续的

        return torch.from_numpy(img), labels_out, self.img_files[index], shapes  # 返回图像、标签、文件名和形状

    @staticmethod
    def collate_fn(batch):
        img, label, path, shapes = zip(*batch)  # 解压缩批次数据，得到图像、标签、路径和形状
        for i, l in enumerate(label):
            l[:, 0] = i  # 为每个标签添加目标图像索引，用于构建目标
        return torch.stack(img, 0), torch.cat(label, 0), path, shapes  # 返回堆叠后的图像、拼接后的标签、路径和形状

    @staticmethod
    def collate_fn4(batch):
        img, label, path, shapes = zip(*batch)  # 解压缩批次数据，得到图像、标签、路径和形状
        n = len(shapes) // 4  # 每个组合的图像数量
        img4, label4, path4, shapes4 = [], [], path[:n], shapes[:n]  # 初始化新列表

        ho = torch.tensor([[0.0, 0, 0, 1, 0, 0]])  # 垂直翻转偏移
        wo = torch.tensor([[0.0, 0, 1, 0, 0, 0]])  # 水平翻转偏移
        s = torch.tensor([[1, 1, 0.5, 0.5, 0.5, 0.5]])  # 缩放因子

        for i in range(n):  # 遍历每组图像
            i *= 4  # 每组包含4张图像
            if random.random() < 0.5:  # 50%概率选择插值
                im = F.interpolate(img[i].unsqueeze(0).float(), scale_factor=2.0, mode='bilinear', align_corners=False)[
                    0].type(img[i].type())  # 使用双线性插值放大图像
                l = label[i]  # 取对应标签
            else:
                # 将四张图像拼接成一张
                im = torch.cat((torch.cat((img[i], img[i + 1]), 1), torch.cat((img[i + 2], img[i + 3]), 1)), 2)
                # 拼接标签并应用偏移和缩放
                l = torch.cat((label[i], label[i + 1] + ho, label[i + 2] + wo, label[i + 3] + ho + wo), 0) * s
            img4.append(im)  # 添加处理后的图像
            label4.append(l)  # 添加处理后的标签

        for i, l in enumerate(label4):
            l[:, 0] = i  # 为每个标签添加目标图像索引，用于构建目标

        return torch.stack(img4, 0), torch.cat(label4, 0), path4, shapes4  # 返回堆叠后的图像、拼接后的标签、路径和形状


# Ancillary functions --------------------------------------------------------------------------------------------------
def load_image(self, i):
    # 从数据集中加载索引为 'i' 的一张图像，返回图像、原始高度宽度和调整后高度宽度
    im = self.imgs[i]  # 从缓存中获取图像
    if im is None:  # 如果未缓存到内存
        npy = self.img_npy[i]  # 获取对应的.npy文件路径
        if npy and npy.exists():  # 如果.npy文件存在，则加载
            im = np.load(npy)
        else:  # 否则，从图像路径读取图像
            path = self.img_files[i]  # 获取图像路径
            im = cv2.imread(path)  # 读取图像 (BGR格式)
            assert im is not None, f'Image Not Found {path}'  # 确保图像成功读取
        h0, w0 = im.shape[:2]  # 获取原始高度和宽度
        r = self.img_size / max(h0, w0)  # 计算缩放比例
        if r != 1:  # 如果尺寸不相等
            im = cv2.resize(im, (int(w0 * r), int(h0 * r)),
                            interpolation=cv2.INTER_AREA if r < 1 and not self.augment else cv2.INTER_LINEAR)  # 调整图像大小
        return im, (h0, w0), im.shape[:2]  # 返回图像、原始高度宽度和调整后高度宽度
    else:
        return self.imgs[i], self.img_hw0[i], self.img_hw[i]  # 如果缓存中有图像，直接返回缓存的图像及其尺寸



def load_mosaic(self, index):
    # 4图像拼接加载器。加载1张图像和3张随机图像到一个4图像的拼接中
    labels4, segments4 = [], []  # 初始化标签和分段列表
    s = self.img_size  # 图像尺寸
    # 随机确定拼接中心的 x 和 y 坐标
    yc, xc = (int(random.uniform(-x, 2 * s + x)) for x in self.mosaic_border)
    indices = [index] + random.choices(self.indices, k=3)  # 随机选择3个额外的图像索引
    random.shuffle(indices)  # 打乱索引顺序

    for i, index in enumerate(indices):
        # 加载图像
        img, _, (h, w) = load_image(self, index)

        # 将图像放置在 img4 中
        if i == 0:  # 左上角
            img4 = np.full((s * 2, s * 2, img.shape[2]), 114, dtype=np.uint8)  # 创建一个基于4个拼接图像的空白图像
            x1a, y1a, x2a, y2a = max(xc - w, 0), max(yc - h, 0), xc, yc  # 大图像的边界
            x1b, y1b, x2b, y2b = w - (x2a - x1a), h - (y2a - y1a), w, h  # 小图像的边界
        elif i == 1:  # 右上角
            x1a, y1a, x2a, y2a = xc, max(yc - h, 0), min(xc + w, s * 2), yc
            x1b, y1b, x2b, y2b = 0, h - (y2a - y1a), min(w, x2a - x1a), h
        elif i == 2:  # 左下角
            x1a, y1a, x2a, y2a = max(xc - w, 0), yc, xc, min(s * 2, yc + h)
            x1b, y1b, x2b, y2b = w - (x2a - x1a), 0, w, min(y2a - y1a, h)
        elif i == 3:  # 右下角
            x1a, y1a, x2a, y2a = xc, yc, min(xc + w, s * 2), min(s * 2, yc + h)
            x1b, y1b, x2b, y2b = 0, 0, min(w, x2a - x1a), min(y2a - y1a, h)

        img4[y1a:y2a, x1a:x2a] = img[y1b:y2b, x1b:x2b]  # 将图像放置到拼接图像的对应位置
        padw = x1a - x1b  # 计算水平填充
        padh = y1a - y1b  # 计算垂直填充

        # 处理标签
        labels, segments = self.labels[index].copy(), self.segments[index].copy()  # 复制标签和分段信息
        if labels.size:
            labels[:, 1:] = xywhn2xyxy(labels[:, 1:], w, h, padw, padh)  # 将归一化的xywh格式转换为像素xyxy格式
            segments = [xyn2xy(x, w, h, padw, padh) for x in segments]  # 转换分段格式
        labels4.append(labels)  # 添加标签
        segments4.extend(segments)  # 添加分段

    # 连接/裁剪标签
    labels4 = np.concatenate(labels4, 0)  # 连接所有标签
    for x in (labels4[:, 1:], *segments4):
        np.clip(x, 0, 2 * s, out=x)  # 在使用 random_perspective() 时裁剪

    # 数据增强
    img4, labels4, segments4 = copy_paste(img4, labels4, segments4, p=self.hyp['copy_paste'])  # 拷贝粘贴增强
    img4, labels4 = random_perspective(img4, labels4, segments4,
                                       degrees=self.hyp['degrees'],
                                       translate=self.hyp['translate'],
                                       scale=self.hyp['scale'],
                                       shear=self.hyp['shear'],
                                       perspective=self.hyp['perspective'],
                                       border=self.mosaic_border)  # 进行随机透视变换

    return img4, labels4  # 返回拼接图像和标签


def load_mosaic9(self, index):
    # 9图像拼接加载器。加载1张图像和8张随机图像到一个9图像的拼接中
    labels9, segments9 = [], []  # 初始化标签和分段列表
    s = self.img_size  # 图像尺寸
    indices = [index] + random.choices(self.indices, k=8)  # 随机选择8个额外的图像索引
    random.shuffle(indices)  # 打乱索引顺序

    for i, index in enumerate(indices):
        # 加载图像
        img, _, (h, w) = load_image(self, index)

        # 将图像放置在 img9 中
        if i == 0:  # 中心位置
            img9 = np.full((s * 3, s * 3, img.shape[2]), 114, dtype=np.uint8)  # 创建一个基于9个拼接图像的空白图像
            h0, w0 = h, w  # 保存原始高度和宽度
            c = s, s, s + w, s + h  # 基础坐标 (xmin, ymin, xmax, ymax)
        elif i == 1:  # 顶部
            c = s, s - h, s + w, s
        elif i == 2:  # 右上角
            c = s + wp, s - h, s + wp + w, s
        elif i == 3:  # 右侧
            c = s + w0, s, s + w0 + w, s + h
        elif i == 4:  # 右下角
            c = s + w0, s + hp, s + w0 + w, s + hp + h
        elif i == 5:  # 底部
            c = s + w0 - w, s + h0, s + w0, s + h0 + h
        elif i == 6:  # 左下角
            c = s + w0 - wp - w, s + h0, s + w0 - wp, s + h0 + h
        elif i == 7:  # 左侧
            c = s - w, s + h0 - h, s, s + h0
        elif i == 8:  # 左上角
            c = s - w, s + h0 - hp - h, s, s + h0 - hp

        padx, pady = c[:2]  # 获取偏移量
        x1, y1, x2, y2 = (max(x, 0) for x in c)  # 分配坐标

        # 处理标签
        labels, segments = self.labels[index].copy(), self.segments[index].copy()  # 复制标签和分段信息
        if labels.size:
            labels[:, 1:] = xywhn2xyxy(labels[:, 1:], w, h, padx, pady)  # 将归一化的xywh格式转换为像素xyxy格式
            segments = [xyn2xy(x, w, h, padx, pady) for x in segments]  # 转换分段格式
        labels9.append(labels)  # 添加标签
        segments9.extend(segments)  # 添加分段

        # 将图像放入拼接图像中
        img9[y1:y2, x1:x2] = img[y1 - pady:, x1 - padx:]  # img9[ymin:ymax, xmin:xmax]
        hp, wp = h, w  # 保存上一个图像的高度和宽度

    # 随机偏移
    yc, xc = (int(random.uniform(0, s)) for _ in self.mosaic_border)  # 拼接中心的 x 和 y 坐标
    img9 = img9[yc:yc + 2 * s, xc:xc + 2 * s]  # 从拼接图像中裁剪

    # 连接/裁剪标签
    labels9 = np.concatenate(labels9, 0)  # 连接所有标签
    labels9[:, [1, 3]] -= xc  # 调整 x 坐标
    labels9[:, [2, 4]] -= yc  # 调整 y 坐标
    c = np.array([xc, yc])  # 中心坐标
    segments9 = [x - c for x in segments9]  # 调整分段坐标

    for x in (labels9[:, 1:], *segments9):
        np.clip(x, 0, 2 * s, out=x)  # 在使用 random_perspective() 时裁剪

    # 数据增强
    img9, labels9 = random_perspective(img9, labels9, segments9,
                                       degrees=self.hyp['degrees'],
                                       translate=self.hyp['translate'],
                                       scale=self.hyp['scale'],
                                       shear=self.hyp['shear'],
                                       perspective=self.hyp['perspective'],
                                       border=self.mosaic_border)  # 进行随机透视变换

    return img9, labels9  # 返回拼接图像和标签


def create_folder(path='./new'):
    # 创建文件夹
    if os.path.exists(path):
        shutil.rmtree(path)  # 删除已有的输出文件夹
    os.makedirs(path)  # 创建新的输出文件夹

def flatten_recursive(path='../datasets/coco128'):
    # 扁平化递归目录，将所有文件移动到顶层
    new_path = Path(path + '_flat')  # 创建新路径
    create_folder(new_path)  # 创建新文件夹
    for file in tqdm(glob.glob(str(Path(path)) + '/**/*.*', recursive=True)):
        # 遍历目录中的所有文件
        shutil.copyfile(file, new_path / Path(file).name)  # 将文件复制到新文件夹



def extract_boxes(path='../datasets/coco128'):  # 从 utils.datasets 导入 *; 提取框
    # 将检测数据集转换为分类数据集，每个类别一个目录
    path = Path(path)  # 图像目录
    shutil.rmtree(path / 'classifier') if (path / 'classifier').is_dir() else None  # 删除已存在的目录
    files = list(path.rglob('*.*'))  # 获取所有文件
    n = len(files)  # 文件总数
    for im_file in tqdm(files, total=n):
        if im_file.suffix[1:] in IMG_FORMATS:
            # 处理图像
            im = cv2.imread(str(im_file))[..., ::-1]  # BGR 转 RGB
            h, w = im.shape[:2]  # 获取图像的高和宽

            # 加载标签
            lb_file = Path(img2label_paths([str(im_file)])[0])  # 获取对应的标签文件
            if Path(lb_file).exists():
                with open(lb_file) as f:
                    lb = np.array([x.split() for x in f.read().strip().splitlines()], dtype=np.float32)  # 读取标签

                for j, x in enumerate(lb):
                    c = int(x[0])  # 类别
                    f = (path / 'classifier') / f'{c}' / f'{path.stem}_{im_file.stem}_{j}.jpg'  # 新文件名
                    if not f.parent.is_dir():
                        f.parent.mkdir(parents=True)  # 创建类别目录

                    b = x[1:] * [w, h, w, h]  # 边界框
                    # b[2:] = b[2:].max()  # 矩形转换为正方形
                    b[2:] = b[2:] * 1.2 + 3  # 扩展边界框
                    b = xywh2xyxy(b.reshape(-1, 4)).ravel().astype(int)  # 转换为 (x1, y1, x2, y2) 格式

                    b[[0, 2]] = np.clip(b[[0, 2]], 0, w)  # 限制边界框在图像内
                    b[[1, 3]] = np.clip(b[[1, 3]], 0, h)
                    assert cv2.imwrite(str(f), im[b[1]:b[3], b[0]:b[2]]), f'box failure in {f}'  # 保存裁剪的图像

def autosplit(path='../datasets/coco128/images', weights=(0.9, 0.1, 0.0), annotated_only=False):
    """ 自动将数据集拆分为训练/验证/测试集，并保存 path/autosplit_*.txt 文件
    使用方法: from utils.datasets import *; autosplit()
    参数
        path:            图像目录的路径
        weights:         训练、验证、测试的权重 (列表或元组)
        annotated_only:  仅使用有标注的图像
    """
    path = Path(path)  # 图像目录
    files = sorted(x for x in path.rglob('*.*') if x.suffix[1:].lower() in IMG_FORMATS)  # 仅获取图像文件
    n = len(files)  # 文件总数
    random.seed(0)  # 设置随机种子以确保可复现性
    indices = random.choices([0, 1, 2], weights=weights, k=n)  # 根据权重分配每个图像到不同的分组

    txt = ['autosplit_train.txt', 'autosplit_val.txt', 'autosplit_test.txt']  # 3 个 txt 文件
    [(path.parent / x).unlink(missing_ok=True) for x in txt]  # 删除已存在的文件

    print(f'从 {path} 自动拆分图像' + ', 仅使用标注的 *.txt 图像' * annotated_only)
    for i, img in tqdm(zip(indices, files), total=n):
        if not annotated_only or Path(img2label_paths([str(img)])[0]).exists():  # 检查是否有标签
            with open(path.parent / txt[i], 'a') as f:
                f.write('./' + img.relative_to(path.parent).as_posix() + '\n')  # 将图像路径写入相应的 txt 文件


def verify_image_label(args):
    # 验证单个图像-标签对
    im_file, lb_file, prefix = args
    nm, nf, ne, nc, msg, segments = 0, 0, 0, 0, '', []  # 统计（缺失、找到、空、损坏），消息，分段
    try:
        # 验证图像
        im = Image.open(im_file)
        im.verify()  # 使用 PIL 验证图像
        shape = exif_size(im)  # 获取图像尺寸
        assert (shape[0] > 9) & (shape[1] > 9), f'图像尺寸 {shape} 小于 10 像素'
        assert im.format.lower() in IMG_FORMATS, f'无效的图像格式 {im.format}'

        if im.format.lower() in ('jpg', 'jpeg'):
            with open(im_file, 'rb') as f:
                f.seek(-2, 2)
                if f.read() != b'\xff\xd9':  # 检查 JPEG 是否损坏
                    ImageOps.exif_transpose(Image.open(im_file)).save(im_file, 'JPEG', subsampling=0, quality=100)
                    msg = f'{prefix}警告: {im_file}: 损坏的 JPEG 已恢复并保存'

        # 验证标签
        if os.path.isfile(lb_file):
            nf = 1  # 找到标签
            with open(lb_file) as f:
                l = [x.split() for x in f.read().strip().splitlines() if len(x)]
                if any([len(x) > 8 for x in l]):  # 判断是否为分段
                    classes = np.array([x[0] for x in l], dtype=np.float32)
                    segments = [np.array(x[1:], dtype=np.float32).reshape(-1, 2) for x in l]  # (类, xy1...)
                    l = np.concatenate((classes.reshape(-1, 1), segments2boxes(segments)), 1)  # (类, xywh)
                l = np.array(l, dtype=np.float32)
            nl = len(l)
            if nl:
                assert l.shape[1] == 5, f'标签需包含 5 列，检测到 {l.shape[1]} 列'
                assert (l >= 0).all(), f'标签值不能为负数 {l[l < 0]}'
                assert (l[:, 1:] <= 1).all(), f'坐标未归一化或超出范围 {l[:, 1:][l[:, 1:] > 1]}'
                _, i = np.unique(l, axis=0, return_index=True)
                if len(i) < nl:  # 检查重复行
                    l = l[i]  # 去除重复项
                    if segments:
                        segments = segments[i]
                    msg = f'{prefix}警告: {im_file}: 移除 {nl - len(i)} 个重复标签'
            else:
                ne = 1  # 标签为空
                l = np.zeros((0, 5), dtype=np.float32)
        else:
            nm = 1  # 标签缺失
            l = np.zeros((0, 5), dtype=np.float32)
        return im_file, l, shape, segments, nm, nf, ne, nc, msg
    except Exception as e:
        nc = 1
        msg = f'{prefix}警告: {im_file}: 忽略损坏的图像/标签: {e}'
        return [None, None, None, None, nm, nf, ne, nc, msg]


def dataset_stats(path='coco128.yaml', autodownload=False, verbose=False, profile=False, hub=False):
    """ 返回数据集统计字典，包括每个类别在每个拆分中的图像和实例计数
    要在父目录中运行：export PYTHONPATH="$PWD/yolov3"
    Usage1: from utils.datasets import *; dataset_stats('coco128.yaml', autodownload=True)
    Usage2: from utils.datasets import *; dataset_stats('../datasets/coco128_with_yaml.zip')
    参数
        path:           data.yaml 或包含 data.yaml 的 data.zip 的路径
        autodownload:   如果本地不存在数据集，则尝试下载数据集
        verbose:        打印统计字典
        profile:        执行性能分析
        hub:            是否进行 HUB 操作，用于网络/应用查看
    """

    def round_labels(labels):
        # 更新标签为整数类和 6 位小数
        return [[int(c), *(round(x, 4) for x in points)] for c, *points in labels]

    def unzip(path):
        # 解压 data.zip，注意：path/to/abc.zip 必须解压到 'path/to/abc/' 中
        if str(path).endswith('.zip'):  # path 是 data.zip
            assert Path(path).is_file(), f'解压 {path} 时出错，文件未找到'
            ZipFile(path).extractall(path=path.parent)  # 解压
            dir = path.with_suffix('')  # 数据集目录 == 压缩包名称
            return True, str(dir), next(dir.rglob('*.yaml'))  # 已压缩，数据目录，yaml 路径
        else:  # path 是 data.yaml
            return False, None, path

    def hub_ops(f, max_dim=1920):
        # HUB 操作用于一个图像 'f'：调整大小并以较低的质量保存到 /dataset-hub 用于网络/应用查看
        f_new = im_dir / Path(f).name  # dataset-hub 图像文件名
        try:  # 使用 PIL
            im = Image.open(f)
            r = max_dim / max(im.height, im.width)  # 比例
            if r < 1.0:  # 图像太大
                im = im.resize((int(im.width * r), int(im.height * r)))
            im.save(f_new, 'JPEG', quality=75, optimize=True)  # 保存
        except Exception as e:  # 使用 OpenCV
            print(f'警告: HUB 操作 PIL 失败 {f}: {e}')
            im = cv2.imread(f)
            im_height, im_width = im.shape[:2]
            r = max_dim / max(im_height, im_width)  # 比例
            if r < 1.0:  # 图像太大
                im = cv2.resize(im, (int(im_width * r), int(im_height * r)), interpolation=cv2.INTER_LINEAR)
            cv2.imwrite(str(f_new), im)

    zipped, data_dir, yaml_path = unzip(Path(path))
    with open(check_yaml(yaml_path), errors='ignore') as f:
        data = yaml.safe_load(f)  # 数据字典
        if zipped:
            data['path'] = data_dir  # TODO: 这应该是 dir.resolve() 吗？
    check_dataset(data, autodownload)  # 如果缺失，下载数据集
    hub_dir = Path(data['path'] + ('-hub' if hub else ''))
    stats = {'nc': data['nc'], 'names': data['names']}  # 统计字典
    for split in 'train', 'val', 'test':
        if data.get(split) is None:
            stats[split] = None  # 例如没有测试集
            continue
        x = []
        dataset = LoadImagesAndLabels(data[split])  # 加载数据集
        for label in tqdm(dataset.labels, total=dataset.n, desc='统计信息'):
            x.append(np.bincount(label[:, 0].astype(int), minlength=data['nc']))
        x = np.array(x)  # shape(128x80)
        stats[split] = {'instance_stats': {'total': int(x.sum()), 'per_class': x.sum(0).tolist()},
                        'image_stats': {'total': dataset.n, 'unlabelled': int(np.all(x == 0, 1).sum()),
                                        'per_class': (x > 0).sum(0).tolist()},
                        'labels': [{str(Path(k).name): round_labels(v.tolist())} for k, v in
                                   zip(dataset.img_files, dataset.labels)]}

        if hub:
            im_dir = hub_dir / 'images'
            im_dir.mkdir(parents=True, exist_ok=True)
            for _ in tqdm(ThreadPool(NUM_THREADS).imap(hub_ops, dataset.img_files), total=dataset.n, desc='HUB 操作'):
                pass

    # 性能分析
    stats_path = hub_dir / 'stats.json'
    if profile:
        for _ in range(1):
            file = stats_path.with_suffix('.npy')
            t1 = time.time()
            np.save(file, stats)
            t2 = time.time()
            x = np.load(file, allow_pickle=True)
            print(f'stats.npy times: 读取 {time.time() - t2:.3f}s, 写入 {t2 - t1:.3f}s')

            file = stats_path.with_suffix('.json')
            t1 = time.time()
            with open(file, 'w') as f:
                json.dump(stats, f)  # 保存 stats *.json
            t2 = time.time()
            with open(file) as f:
                x = json.load(f)  # 加载 hyps 字典
            print(f'stats.json times: 读取 {time.time() - t2:.3f}s, 写入 {t2 - t1:.3f}s')

    # 保存、打印并返回
    if hub:
        print(f'保存 {stats_path.resolve()}...')
        with open(stats_path, 'w') as f:
            json.dump(stats, f)  # 保存 stats.json
    if verbose:
        print(json.dumps(stats, indent=2, sort_keys=False))
    return stats

