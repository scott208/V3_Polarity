# YOLOv3 🚀 by Ultralytics, GPL-3.0 license
"""
General utils
"""

import contextlib
import glob
import logging
import math
import os
import platform
import random
import re
import shutil
import signal
import time
import urllib
from itertools import repeat
from multiprocessing.pool import ThreadPool
from pathlib import Path
from subprocess import check_output
from zipfile import ZipFile

import cv2
import numpy as np
import pandas as pd
import pkg_resources as pkg
import torch
import torchvision
import yaml

from utils.downloads import gsutil_getsize
from utils.metrics import box_iou, fitness

# Settings

# 设置 PyTorch 的打印选项：行宽为 320，精度为小数点后 5 位，并使用 'long' 格式文件打印
torch.set_printoptions(linewidth=320, precision=5, profile='long')
# 设置 NumPy 的打印选项：行宽为 320，小数点格式为 '%11.5g'（即小数点后 5 位，有效数字为 11 位）
np.set_printoptions(linewidth=320, formatter={'float_kind': '{:11.5g}'.format})  # format short g, %precision=5
# 设置 pandas 显示选项：最多显示 10 列
pd.options.display.max_columns = 10
# 设置 OpenCV 使用的线程数为 0，防止其与 PyTorch DataLoader 的多线程不兼容问题
cv2.setNumThreads(0)  # prevent OpenCV from multithreading (incompatible with PyTorch DataLoader)
# 设置环境变量 'NUMEXPR_MAX_THREADS' 为当前 CPU 核心数与 8 之间的最小值，最大线程数不超过 8
os.environ['NUMEXPR_MAX_THREADS'] = str(min(os.cpu_count(), 8))  # NumExpr max threads

FILE = Path(__file__).resolve()
ROOT = FILE.parents[1]  # root directory


def set_logging(name=None, verbose=True):
    # 设置日志记录的级别并返回日志记录器
    rank = int(os.getenv('RANK', -1))  # 获取环境变量 'RANK' 的值，如果未设置则默认为 -1，用于多 GPU 训练中的排名
    # 配置日志的基本设置：消息格式为简单的 "%(message)s"
    # 如果 verbose 为 True 并且 rank 为 -1 或 0（即单 GPU 或主进程），则日志级别为 INFO，否则为 WARNING
    logging.basicConfig(format="%(message)s", level=logging.INFO if (verbose and rank in (-1, 0)) else logging.WARNING)
    return logging.getLogger(name)  # 返回配置好的日志记录器
# # 定义全局日志记录器（在 train.py、val.py、detect.py 等模块中使用）
LOGGER = set_logging(__name__)


class Profile(contextlib.ContextDecorator):
    # 用法：可以用作 @Profile() 装饰器或 'with Profile():' 上下文管理器
    def __enter__(self):
        # 进入上下文时记录开始时间
        self.start = time.time()

    def __exit__(self, type, value, traceback):
        # 退出上下文时计算并打印所耗时间
        print(f'Profile results: {time.time() - self.start:.5f}s')



class Timeout(contextlib.ContextDecorator):
    # 用法：可以用作 @Timeout(seconds) 装饰器或 'with Timeout(seconds):' 上下文管理器
    def __init__(self, seconds, *, timeout_msg='', suppress_timeout_errors=True):
        self.seconds = int(seconds)  # 设置超时时间（秒）
        self.timeout_message = timeout_msg  # 设置超时信息
        self.suppress = bool(suppress_timeout_errors)  # 是否抑制超时错误

    def _timeout_handler(self, signum, frame):
        # 超时处理程序，抛出 TimeoutError 并显示超时信息
        raise TimeoutError(self.timeout_message)

    def __enter__(self):
        # 进入上下文时设置信号处理器为 _timeout_handler，开始倒计时
        signal.signal(signal.SIGALRM, self._timeout_handler)  # 设置 SIGALRM 的处理程序
        signal.alarm(self.seconds)  # 开始倒计时，触发 SIGALRM

    def __exit__(self, exc_type, exc_val, exc_tb):
        # 退出上下文时取消任何已调度的 SIGALRM
        signal.alarm(0)  # 取消调度的 SIGALRM
        if self.suppress and exc_type is TimeoutError:  # 如果抑制超时错误并且发生了 TimeoutError
            return True  # 抑制 TimeoutError

class WorkingDirectory(contextlib.ContextDecorator):
    # 用法：可以用作 @WorkingDirectory(dir) 装饰器或 'with WorkingDirectory(dir):' 上下文管理器
    def __init__(self, new_dir):
        self.dir = new_dir  # 目标目录
        self.cwd = Path.cwd().resolve()  # 当前工作目录

    def __enter__(self):
        # 进入上下文时切换到目标目录
        os.chdir(self.dir)

    def __exit__(self, exc_type, exc_val, exc_tb):
        # 退出上下文时切换回原工作目录
        os.chdir(self.cwd)

def try_except(func):
    # try-except 函数。用法：@try_except 装饰器
    def handler(*args, **kwargs):
        try:
            # 尝试执行被装饰的函数
            func(*args, **kwargs)
        except Exception as e:
            # 捕获任何异常并打印异常信息
            print(e)
    return handler


def methods(instance):
    # 获取类或实例的方法
    return [f for f in dir(instance) if callable(getattr(instance, f)) and not f.startswith("__")]

# 打印参数的功能
def print_args(name, opt):
    # 打印命令行解析器的参数
    # LOGGER.info(colorstr(f'{name}: ') + ', '.join(f'{k}={v}' for k, v in vars(opt).items()))
    print(colorstr(f'{name}: ') + ', '.join(f'{k}={v}' for k, v in vars(opt).items()))

def init_seeds(seed=0):
    # 初始化随机数生成器（RNG）的种子 https://pytorch.org/docs/stable/notes/randomness.html
    # cudnn seed 0 设置更慢但更可重复，否则更快但较不可重复
    import torch.backends.cudnn as cudnn
    random.seed(seed)  # 设置 Python 随机种子
    np.random.seed(seed)  # 设置 NumPy 随机种子
    torch.manual_seed(seed)  # 设置 PyTorch 随机种子
    # 根据种子设置 cudnn 的 benchmark 和 deterministic 属性
    cudnn.benchmark, cudnn.deterministic = (False, True) if seed == 0 else (True, False)

def intersect_dicts(da, db, exclude=()):
    # 返回两个字典中键和值匹配的交集字典，排除 'exclude' 中的键，并使用 da 的值
    return {k: v for k, v in da.items() if k in db and not any(x in k for x in exclude) and v.shape == db[k].shape}

def get_latest_run(search_dir='.'):
    # 返回 /runs 目录中最近的 'last.pt' 文件路径（用于恢复训练）
    last_list = glob.glob(f'{search_dir}/**/last*.pt', recursive=True)  # 查找所有匹配的 'last*.pt' 文件
    return max(last_list, key=os.path.getctime) if last_list else ''  # 返回最新的文件路径，如果没有则返回空字符串

def user_config_dir(dir='Ultralytics', env_var='YOLOV3_CONFIG_DIR'):
    # 返回用户配置目录的路径。如果环境变量存在，则优先使用环境变量的值。如果需要则创建该目录。
    env = os.getenv(env_var)  # 获取环境变量值
    if env:
        path = Path(env)  # 使用环境变量指定的路径
    else:
        # 不同操作系统的默认配置目录
        cfg = {'Windows': 'AppData/Roaming', 'Linux': '.config', 'Darwin': 'Library/Application Support'}
        path = Path.home() / cfg.get(platform.system(), '')  # 获取操作系统特定的配置目录
        # 如果目录不可写，则使用 /tmp 目录
        path = (path if is_writeable(path) else Path('/tmp')) / dir  # GCP 和 AWS lambda 修复，只有 /tmp 可写
    path.mkdir(exist_ok=True)  # 如果目录不存在则创建
    return path  # 返回配置目录路径


def is_writeable(dir, test=False):
    # 如果目录具有写权限则返回 True，如果 test=True 则测试打开一个具有写权限的文件
    if test:  # 方法 1
        file = Path(dir) / 'tmp.txt'
        try:
            with open(file, 'w'):  # 以写权限打开文件
                pass
            file.unlink()  # 删除文件
            return True
        except OSError:
            return False
    else:  # 方法 2
        return os.access(dir, os.R_OK)  # 在 Windows 上可能存在问题

def is_docker():
    # 判断当前环境是否为 Docker 容器
    return Path('/workspace').exists()  # 或者 Path('/.dockerenv').exists()

def is_colab():
    # 判断当前环境是否为 Google Colab 实例
    try:
        import google.colab
        return True
    except ImportError:
        return False

def is_pip():
    # 判断文件是否在 pip 包中
    return 'site-packages' in Path(__file__).resolve().parts

def is_ascii(s=''):
    # 判断字符串是否由所有 ASCII（无 UTF）字符组成
    # 注意：str().isascii() 在 Python 3.7 中引入
    s = str(s)  # 将列表、元组、None 等转换为字符串
    return len(s.encode().decode('ascii', 'ignore')) == len(s)

def is_chinese(s='人工智能'):
    # 判断字符串是否包含任何中文字符
    return re.search('[\u4e00-\u9fff]', s)

def emojis(str=''):
    # 返回平台相关的表情符号安全版本的字符串
    return str.encode().decode('ascii', 'ignore') if platform.system() == 'Windows' else str

def file_size(path):
    # 返回文件/目录的大小（以 MB 为单位）
    path = Path(path)
    if path.is_file():
        # 如果路径是文件，则返回文件大小（MB）
        return path.stat().st_size / 1E6
    elif path.is_dir():
        # 如果路径是目录，则返回目录内所有文件的总大小（MB）
        return sum(f.stat().st_size for f in path.glob('**/*') if f.is_file()) / 1E6
    else:
        # 如果路径既不是文件也不是目录，则返回 0.0
        return 0.0

def check_online():
    """
        检查互联网连接。

        尝试创建到已知可靠服务器的套接字连接。

        参数:
        timeout (int): 连接尝试的超时时间（秒）。

        返回:
        bool: 在线时返回 True，否则返回 False。
        """
    import socket
    try:
        # 尝试创建到可靠服务器的套接字连接
        socket.create_connection(("1.1.1.1", 443), 5)  # check host accessibility
        return True
    except OSError:
        return False


@try_except
@WorkingDirectory(ROOT)
def check_git_status():
    # 如果代码过期，推荐使用 'git pull'
    msg = ', for updates see https://github.com/ultralytics/yolov3'
    print(colorstr('github: '), end='')
    assert Path('.git').exists(), 'skipping check (not a git repository)' + msg  # 确保当前目录是一个 git 仓库
    assert not is_docker(), 'skipping check (Docker image)' + msg # 确保代码不是运行在 Docker 容器中
    assert check_online(), 'skipping check (offline)' + msg # 确保系统在线
    # 获取最新更改并获取远程仓库的 URL
    cmd = 'git fetch && git config --get remote.origin.url'
    url = check_output(cmd, shell=True, timeout=5).decode().strip().rstrip('.git')  # git fetch
    branch = check_output('git rev-parse --abbrev-ref HEAD', shell=True).decode().strip()  # 获取当前分支名称
    n = int(check_output(f'git rev-list {branch}..origin/master --count', shell=True))  # 计算本地分支落后于远程主分支的提交数量
    if n > 0:
        s = f"⚠️ YOLOv3 is out of date by {n} commit{'s' * (n > 1)}. Use `git pull` or `git clone {url}` to update."  # 如果有更新，建议拉取最新更改
    else:
        s = f'up to date with {url} ✅'  # 如果已经是最新，显示同步信息
    print(emojis(s))  # emoji-safe   # 打印带有表情符号的状态信息


def check_python(minimum='3.6.2'):
    """
            check_python是检查当前的版本号是否满足最小版本号minimum
            被调用：函数check_requirements中
        """
    # 对比当前版本号和输出的至少的版本号(python版本一般是向下兼容的)
    # 如果满足返回result=True 反正返回result=False
    # pkg.parse_version(版本号)用于对比两个版本号的大小
    check_version(platform.python_version(), minimum, name='Python ', hard=True)


def check_version(current='0.0.0', minimum='0.0.0', name='version ', pinned=False, hard=False):
    # Check version vs. required version
    current, minimum = (pkg.parse_version(x) for x in (current, minimum))
    result = (current == minimum) if pinned else (current >= minimum)  # bool
    if hard:  # assert min requirements met
        assert result, f'{name}{minimum} required by YOLOv3, but {name}{current} is currently installed'
    else:
        return result


@try_except
def check_requirements(requirements=ROOT / 'requirements.txt', exclude=(), install=True):
    """
        检查已安装的依赖项是否符合要求，并可选地安装缺失的依赖项。

        Args:
            requirements (str or Path, optional): requirements.txt 文件的路径或包名称的列表/元组。默认为 ROOT / 'requirements.txt'。
            exclude (tuple, optional): 要排除检查或安装的特定依赖项。
            install (bool, optional): 是否尝试自动安装缺失的依赖项。

        Returns:
            None

        Raises:
            AssertionError: 如果指定的 requirements 文件不存在。

        """
    # 设置带颜色的日志前缀
    prefix = colorstr('red', 'bold', 'requirements:')
    check_python()  # 检查 Python 版本

    # 解析requirements.txt中的所有包 解析成list 里面存放着一个个的pkg_resources.Requirement类
    # 如: ['matplotlib>=3.2.2', 'numpy>=1.18.5', ……]
    if isinstance(requirements, (str, Path)):  # requirements.txt file
        file = Path(requirements)  # 将str字符串requirements转换成路径requirements
        assert file.exists(), f"{prefix} {file.resolve()} not found, check failed."
        with file.open() as f:
            # pkg_resources.parse_requirements:可以解析file中的每一条要求
            # 每一行转换为pkg_resources.Requirement类并进行进一步处理
            # 处理形式为调用每一行对应的name和specifier属性。前者代表需要包的名称，后者代表版本
            # 返回list 每个元素是requirements.txt的一行 如: ['matplotlib>=3.2.2', 'numpy>=1.18.5', ……]
            requirements = [f'{x.name}{x.specifier}' for x in pkg.parse_requirements(f) if x.name not in exclude]
    else:  # list or tuple of packages
        requirements = [x for x in requirements if x not in exclude]

    n = 0  # 统计下面程序更新包的个数 number of packages updates
    for r in requirements:  # 依次检查环境中安装的包(及每个包对应的依赖包)是否满足requirements中的每一个最低要求安装包
        try:
            pkg.require(r)  # pkg_resources.require(file) 返回对应包所需的所有依赖包 当这些包有哪个未安装或者版本不对的时候就会报错
        except Exception as e:
            s = f"{prefix} {r} not found and is required by YOLOv3"
            if install:
                print(f"{s}, attempting auto-update...")
                try:
                    assert check_online(), f"'pip install {r}' skipped (offline)"
                    print(check_output(f"pip install '{r}'", shell=True).decode())
                    n += 1
                except Exception as e:
                    print(f'{prefix} {e}')
            else:
                print(f'{s}. Please install and rerun your command.')

    if n:  # if packages updated
        # if packages updated 打印一些更新信息
        source = file.resolve() if 'file' in locals() else requirements
        s = f"{prefix} {n} package{'s' * (n > 1)} updated per {source}\n" \
            f"{prefix} ⚠️ {colorstr('bold', 'Restart runtime or rerun command for updates to take effect')}\n"
        print(emojis(s))


def check_img_size(imgsz, s=32, floor=0):
    # 验证图像尺寸在每个维度上是否为步幅 s 的倍数
    # imgsz 可以是整数（例如 img_size=640）或列表（例如 img_size=[640, 480]）
    if isinstance(imgsz, int):  # 如果 imgsz 是整数
        new_size = max(make_divisible(imgsz, int(s)), floor)  # 将尺寸调整为步幅 s 的倍数，并不低于 floor
    else:  # 如果 imgsz 是列表
        new_size = [max(make_divisible(x, int(s)), floor) for x in imgsz]  # 对列表中的每个尺寸进行相同的调整
    # 如果调整后的尺寸与原始尺寸不同，打印警告信息
    if new_size != imgsz:
        print(f'WARNING: --img-size {imgsz} must be multiple of max stride {s}, updating to {new_size}')
    return new_size  # 返回调整后的尺寸


def check_imshow():
    # 检查环境是否支持图像显示
    try:
        # 确保不在 Docker 环境中
        assert not is_docker(), 'cv2.imshow() 在 Docker 环境中被禁用'
        # 确保不在 Google Colab 环境中
        assert not is_colab(), 'cv2.imshow() 在 Google Colab 环境中被禁用'
        # 尝试使用 OpenCV 显示一个测试图像
        cv2.imshow('test', np.zeros((1, 1, 3)))  # 显示一个 1x1 像素的黑色图像
        cv2.waitKey(1)  # 等待 1 毫秒以处理显示
        cv2.destroyAllWindows()  # 关闭所有 OpenCV 窗口
        cv2.waitKey(1)  # 等待 1 毫秒以确保窗口关闭
        return True  # 如果没有异常，则环境支持图像显示
    except Exception as e:
        # 捕获任何异常并打印警告信息
        print(f'WARNING: 环境不支持 cv2.imshow() 或 PIL Image.show() 图像显示\n{e}')
        return False  # 环境不支持图像显示

# 总的来说，这个函数确保了文件的后缀符合指定的格式要求，并在格式不符时提供明确的错误信息。
def check_suffix(file='yolov3.pt', suffix=('.pt',), msg=''):
    # 检查文件的后缀是否可接受
    if file and suffix:
        if isinstance(suffix, str):
            suffix = [suffix]
        for f in file if isinstance(file, (list, tuple)) else [file]:
            s = Path(f).suffix.lower()  # 使用 Path 对象获取文件的后缀，并将其转换为小写。
            if len(s):
                assert s in suffix, f"{msg}{f} acceptable suffix is {suffix}"  # 检查后缀是否在允许的后缀列表中，如果不在，则抛出带有自定义消息的断言错误。

def check_yaml(file, suffix=('.yaml', '.yml')):
    #  搜索/下载YAML文件（如果有必要）并返回路径，检查后缀
    return check_file(file, suffix)

# 首先检查文件是否已经存在，如果存在则直接返回文件路径。
# 如果 file 不是 URL，则在预定义的目录中搜索文件，并返回找到的唯一文件路径。
def check_file(file, suffix=''):
    check_suffix(file, suffix)  # 调用check_suffix 函数，用于检查文件后缀是否符合要求
    file = str(file)  # 将file转换为字符串格式。
    if Path(file).is_file() or file == '':  # 检查文件是否已经存在或者file是空字符
        return file  #  如果文件已经存在，直接返回 file。

    # 如果 file 是以 http:/ 或 https:/ 开头的字符串，表示需要下载文件。
    # 本质上我们都会提供数据，另外利用该段代码进行数据或者权重的下载容易导致失败，因此可忽略该段代码
    elif file.startswith(('http:/', 'https:/')):  # download
        url = str(Path(file)).replace(':/', '://')  # Pathlib turns :// -> :/
        file = Path(urllib.parse.unquote(file).split('?')[0]).name  # '%2F' to '/', split https://url.com/file.txt?auth
        if Path(file).is_file():
            print(f'Found {url} locally at {file}')  # file already exists
        else:
            print(f'Downloading {url} to {file}...')
            torch.hub.download_url_to_file(url, file)
            assert Path(file).exists() and Path(file).stat().st_size > 0, f'File download failed: {url}'  # check
        return file


    else:  # 如果 file不是URL，那么搜索本地文件。
        files = []
        for d in 'data', 'models', 'utils':  # 在 data, models, utils 目录下使用 glob.glob 搜索符合条件的文件：
            files.extend(glob.glob(str(ROOT / d / '**' / file), recursive=True))  # find file
        assert len(files), f'File not found: {file}'  # 断言找到了唯一的文件
        assert len(files) == 1, f"Multiple files match '{file}', specify exact path: {files}"  # assert unique
        return files[0]  # 返回找到的文件路径

def check_dataset(data, autodownload=True):
    # 如果数据集在本地未找到，则下载和/或解压数据集
    # 用法示例: https://github.com/ultralytics/yolov5/releases/download/v1.0/coco128_with_yaml.zip
    # 下载（可选）
    extract_dir = ''
    if isinstance(data, (str, Path)) and str(data).endswith('.zip'):  # 如果数据集路径以 .zip 结尾（例如 gs://bucket/dir/coco128.zip）
        download(data, dir='../datasets', unzip=True, delete=False, curl=False, threads=1)  # 下载并解压
        # 查找解压后的 YAML 文件
        data = next((Path('../datasets') / Path(data).stem).rglob('*.yaml'))
        extract_dir, autodownload = data.parent, False  # 设置解压目录并禁用自动下载

    # 读取 YAML 文件（可选）
    if isinstance(data, (str, Path)):
        with open(data, errors='ignore') as f:
            data = yaml.safe_load(f)  # 读取 YAML 文件并解析为字典

    # 解析 YAML 文件中的数据
    path = extract_dir or Path(data.get('path') or '')  # 获取数据集路径，默认为当前目录 '.'
    for k in 'train', 'val', 'test':
        if data.get(k):  # 如果 'train'、'val' 或 'test' 键存在
            # 将路径前缀添加到数据路径中
            data[k] = str(path / data[k]) if isinstance(data[k], str) else [str(path / x) for x in data[k]]

    # 确保数据集中包含 'nc' 键
    assert 'nc' in data, "Dataset 'nc' key missing."
    # 如果数据集中缺少 'names' 键，则为每个类生成默认名称
    if 'names' not in data:
        data['names'] = [f'class{i}' for i in range(data['nc'])]

    # 获取训练、验证、测试路径和下载信息
    train, val, test, s = (data.get(x) for x in ('train', 'val', 'test', 'download'))
    # 验证 'val' 路径是否存在
    if val:
        val = [Path(x).resolve() for x in (val if isinstance(val, list) else [val])]  # 解析 'val' 路径
        if not all(x.exists() for x in val):  # 如果 'val' 路径中的任何路径不存在
            print('\nWARNING: Dataset not found, nonexistent paths: %s' % [str(x) for x in val if not x.exists()])
            if s and autodownload:  # 如果需要下载脚本并且启用了自动下载
                root = path.parent if 'path' in data else '..'  # 设置解压目录
                if s.startswith('http') and s.endswith('.zip'):  # 如果下载地址是 URL 并以 .zip 结尾
                    f = Path(s).name  # 获取文件名
                    print(f'Downloading {s} to {f}...')
                    torch.hub.download_url_to_file(s, f)  # 下载文件
                    Path(root).mkdir(parents=True, exist_ok=True)  # 创建目录
                    ZipFile(f).extractall(path=root)  # 解压文件
                    Path(f).unlink()  # 删除 ZIP 文件
                    r = None  # 下载成功
                elif s.startswith('bash '):  # 如果下载地址是 bash 脚本
                    print(f'Running {s} ...')
                    r = os.system(s)  # 执行 bash 脚本
                else:  # 如果下载地址是 python 脚本
                    r = exec(s, {'yaml': data})  # 执行 python 脚本
                print(f"Dataset autodownload {f'success, saved to {root}' if r in (0, None) else 'failure'}\n")
            else:
                raise Exception('Dataset not found.')  # 如果未找到数据集且不启用自动下载，则引发异常
    return data  # 返回数据字典



def url2file(url):
    # 将 URL 转换为文件名，例如 https://url.com/file.txt?auth -> file.txt
    url = str(Path(url)).replace(':/', '://')  # Pathlib 处理 URL 时将 :// 转换为 :/
    file = Path(urllib.parse.unquote(url)).name.split('?')[0]  # 解码 URL，将 '%2F' 转换为 '/'，然后去除查询参数部分
    return file  # 返回文件名

def download(url, dir='.', unzip=True, delete=True, curl=False, threads=1):
    # 多线程文件下载和解压函数，用于 data.yaml 的自动下载
    def download_one(url, dir):
        # 下载单个文件
        f = dir / Path(url).name  # 生成文件名
        if Path(url).is_file():  # 如果文件在当前路径中存在
            Path(url).rename(f)  # 移动到目标目录
        elif not f.exists():  # 如果目标文件不存在
            print(f'Downloading {url} to {f}...')
            if curl:
                # 使用 curl 下载，支持重试和断点续传
                os.system(f"curl -L '{url}' -o '{f}' --retry 9 -C -")
            else:
                # 使用 torch.hub 下载
                torch.hub.download_url_to_file(url, f, progress=True)
        # 如果需要解压且文件后缀为 .zip 或 .gz
        if unzip and f.suffix in ('.zip', '.gz'):
            print(f'Unzipping {f}...')
            if f.suffix == '.zip':
                # 解压 .zip 文件
                ZipFile(f).extractall(path=dir)
            elif f.suffix == '.gz':
                # 解压 .gz 文件
                os.system(f'tar xfz {f} --directory {f.parent}')
            if delete:
                # 删除原始压缩文件
                f.unlink()
    dir = Path(dir)
    dir.mkdir(parents=True, exist_ok=True)  # 创建目录（如果不存在）
    if threads > 1:
        # 使用多线程下载
        pool = ThreadPool(threads)
        pool.imap(lambda x: download_one(*x), zip(url, repeat(dir)))  # 多线程下载文件
        pool.close()
        pool.join()
    else:
        # 单线程下载
        for u in [url] if isinstance(url, (str, Path)) else url:
            download_one(u, dir)

# make_divisible 函数的作用是将输入x调整为大于或等于x且能被divisor整除的最小整数。
# 例如math.ceil(1.875) 等于 2
def make_divisible(x, divisor):
    # Returns x evenly divisible by divisor
    return math.ceil(x / divisor) * divisor

def clean_str(s):
    # 清理字符串，通过将特殊字符替换为下划线 _
    return re.sub(pattern="[|@#!¡·$€%&()=?¿^*;:,¨´><+]", repl="_", string=s)

def one_cycle(y1=0.0, y2=1.0, steps=100):
    # 返回一个 lambda 函数，用于从 y1 到 y2 的正弦波形 ramp（见 https://arxiv.org/pdf/1812.01187.pdf）
    # 该函数基于给定的步骤数 steps 创建一个从 y1 到 y2 的周期性变化
    return lambda x: ((1 - math.cos(x * math.pi / steps)) / 2) * (y2 - y1) + y1

def colorstr(*input):
    # 为字符串添加颜色样式 https://en.wikipedia.org/wiki/ANSI_escape_code，例如 colorstr('blue', 'hello world')
    # *input 允许传入多个参数，第一个或多个是颜色样式，最后一个是字符串内容
    *args, string = input if len(input) > 1 else ('blue', 'bold', input[0])  # 解析颜色样式和字符串内容
    colors = {
        'black': '\033[30m',  # 基本颜色
        'red': '\033[31m',
        'green': '\033[32m',
        'yellow': '\033[33m',
        'blue': '\033[34m',
        'magenta': '\033[35m',
        'cyan': '\033[36m',
        'white': '\033[37m',
        'bright_black': '\033[90m',  # 亮色
        'bright_red': '\033[91m',
        'bright_green': '\033[92m',
        'bright_yellow': '\033[93m',
        'bright_blue': '\033[94m',
        'bright_magenta': '\033[95m',
        'bright_cyan': '\033[96m',
        'bright_white': '\033[97m',
        'end': '\033[0m',  # 结束颜色样式
        'bold': '\033[1m',  # 粗体
        'underline': '\033[4m'  # 下划线
    }
    # 根据传入的颜色样式构建 ANSI 颜色码字符串，并将其与内容字符串连接，最后加上重置颜色的码
    return ''.join(colors[x] for x in args) + f'{string}' + colors['end']

def labels_to_class_weights(labels, nc=80):
    # 从训练标签中获取类别权重（反向频率）
    if labels[0] is None:  # 如果没有加载标签
        return torch.Tensor()

    labels = np.concatenate(labels, 0)  # 将所有标签合并成一个数组，形状为 (866643, 5)（例如 COCO 数据集）
    classes = labels[:, 0].astype(np.int)  # 提取类别列，labels = [类别 xywh]
    weights = np.bincount(classes, minlength=nc)  # 计算每个类别的出现次数

    # 前置网格点计数（用于 uCE 训练）
    # gpi = ((320 / 32 * np.array([1, 2, 4])) ** 2 * 3).sum()  # 每张图像的网格点数
    # weights = np.hstack([gpi * len(labels)  - weights.sum() * 9, weights * 9]) ** 0.5  # 将网格点数前置到开始位置

    weights[weights == 0] = 1  # 将空的类别权重替换为 1
    weights = 1 / weights  # 计算每个类别的目标数量的倒数
    weights /= weights.sum()  # 归一化
    return torch.from_numpy(weights)  # 转换为 PyTorch 张量并返回


def labels_to_image_weights(labels, nc=80, class_weights=np.ones(80)):
    # 根据类别权重和图像内容生成图像权重
    class_counts = np.array([np.bincount(x[:, 0].astype(np.int), minlength=nc) for x in labels])
    image_weights = (class_weights.reshape(1, nc) * class_counts).sum(1)
    # index = random.choices(range(n), weights=image_weights, k=1)  # 根据图像权重进行样本选择
    return image_weights  # 返回图像权重


def coco80_to_coco91_class():
    # 将 COCO 数据集中的 80 类索引（val2014）转换为 91 类索引（论文中使用的索引）
    # 参考资料: https://tech.amikelive.com/node-718/what-object-categories-labels-are-in-coco-dataset/
    # 以下是从 80 类映射到 91 类的索引转换
    # a = np.loadtxt('data/coco.names', dtype='str', delimiter='\n')
    # b = np.loadtxt('data/coco_paper.names', dtype='str', delimiter='\n')
    # x1 = [list(a[i] == b).index(True) + 1 for i in range(80)]  # darknet 到 coco 的映射
    # x2 = [list(b[i] == a).index(True) if any(b[i] == a) else None for i in range(91)]  # coco 到 darknet 的映射

    # 从 COCO 80 类到 COCO 91 类的映射列表
    x = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 27, 28, 31, 32, 33, 34,
         35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63,
         64, 65, 67, 70, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 84, 85, 86, 87, 88, 89, 90]
    return x  # 返回 COCO 80 类到 COCO 91 类的映射列表


def xyxy2xywh(x):
    # 将边界框从 [x1, y1, x2, y2] 格式转换为 [x, y, w, h] 格式
    # 其中 (x1, y1) 是左上角坐标，(x2, y2) 是右下角坐标

    # 克隆输入以避免修改原始数据，支持 PyTorch 张量和 NumPy 数组
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    # 计算中心坐标 (x, y) 和宽度 (w), 高度 (h)
    y[:, 0] = (x[:, 0] + x[:, 2]) / 2  # x 中心坐标
    y[:, 1] = (x[:, 1] + x[:, 3]) / 2  # y 中心坐标
    y[:, 2] = x[:, 2] - x[:, 0]  # 宽度
    y[:, 3] = x[:, 3] - x[:, 1]  # 高度
    return y  # 返回转换后的边界框

def xywh2xyxy(x):
    # 将边界框从 [x, y, w, h] 格式转换为 [x1, y1, x2, y2] 格式
    # 其中 (x, y) 是中心坐标，w 是宽度，h 是高度
    # 转换后的格式中 (x1, y1) 是左上角坐标，(x2, y2) 是右下角坐标
    # 克隆输入以避免修改原始数据，支持 PyTorch 张量和 NumPy 数组
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    # 计算左上角 (x1, y1) 和右下角 (x2, y2) 的坐标
    y[:, 0] = x[:, 0] - x[:, 2] / 2  # 左上角 x 坐标
    y[:, 1] = x[:, 1] - x[:, 3] / 2  # 左上角 y 坐标
    y[:, 2] = x[:, 0] + x[:, 2] / 2  # 右下角 x 坐标
    y[:, 3] = x[:, 1] + x[:, 3] / 2  # 右下角 y 坐标
    return y  # 返回转换后的边界框

def xywhn2xyxy(x, w=640, h=640, padw=0, padh=0):
    # 将标准化边界框 [x, y, w, h] 转换为 [x1, y1, x2, y2] 格式
    # 其中 (x, y) 是相对于图像的中心坐标，w 和 h 是相对于图像的宽度和高度
    # 转换后的格式中 (x1, y1) 是左上角坐标，(x2, y2) 是右下角坐标
    # w 和 h 是图像的宽度和高度，padw 和 padh 是宽度和高度的填充量
    # 克隆输入以避免修改原始数据，支持 PyTorch 张量和 NumPy 数组
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    # 计算左上角 (x1, y1) 和右下角 (x2, y2) 的坐标
    y[:, 0] = w * (x[:, 0] - x[:, 2] / 2) + padw  # 左上角 x 坐标
    y[:, 1] = h * (x[:, 1] - x[:, 3] / 2) + padh  # 左上角 y 坐标
    y[:, 2] = w * (x[:, 0] + x[:, 2] / 2) + padw  # 右下角 x 坐标
    y[:, 3] = h * (x[:, 1] + x[:, 3] / 2) + padh  # 右下角 y 坐标
    return y  # 返回转换后的边界框


def xyxy2xywhn(x, w=640, h=640, clip=False, eps=0.0):
    # 将边界框从 [x1, y1, x2, y2] 格式转换为 [x, y, w, h] 格式，其中 x 和 y 是中心坐标，w 和 h 是宽度和高度
    # 转换后的格式是标准化的，即相对于图像的宽度和高度
    # clip: 是否将坐标剪裁到 [0, 1] 范围内
    # eps: 一个小的常数，用于防止数值不稳定
    if clip:
        # 如果 clip 为 True，则对坐标进行剪裁，以确保它们在 [0, 1] 范围内
        clip_coords(x, (h - eps, w - eps))  # 注意：此操作会就地修改 x
    # 克隆输入以避免修改原始数据，支持 PyTorch 张量和 NumPy 数组
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    # 计算中心坐标 (x, y) 和宽度 (w) 高度 (h)
    y[:, 0] = ((x[:, 0] + x[:, 2]) / 2) / w  # 中心 x 坐标
    y[:, 1] = ((x[:, 1] + x[:, 3]) / 2) / h  # 中心 y 坐标
    y[:, 2] = (x[:, 2] - x[:, 0]) / w  # 宽度
    y[:, 3] = (x[:, 3] - x[:, 1]) / h  # 高度
    return y  # 返回转换后的边界框


def xyn2xy(x, w=640, h=640, padw=0, padh=0):
    # 将归一化的线段坐标转换为像素坐标，输入形状为 (n, 2)
    # x: 归一化的线段坐标数组或张量，包含 (x_center, y_center) 的归一化值
    # w: 图像宽度
    # h: 图像高度
    # padw: 水平偏移量
    # padh: 垂直偏移量

    # 克隆输入以避免修改原始数据，支持 PyTorch 张量和 NumPy 数组
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    # 将归一化的坐标转换为像素坐标
    y[:, 0] = w * x[:, 0] + padw  # 计算像素坐标 x
    y[:, 1] = h * x[:, 1] + padh  # 计算像素坐标 y

    return y  # 返回像素坐标的数组或张量


def segment2box(segment, width=640, height=640):
    # 将一个分段标签转换为一个框标签，并应用图像内部约束，即将 (xy1, xy2, ...) 转换为 (xyxy)
    # segment: 包含分段的 x 和 y 坐标的数组或张量
    # width: 图像的宽度
    # height: 图像的高度

    # 提取 x 和 y 坐标
    x, y = segment.T  # x 和 y 坐标的转置，假设 segment 的形状是 (n, 2)

    # 约束在图像内部的点
    inside = (x >= 0) & (y >= 0) & (x <= width) & (y <= height)  # 确保坐标在图像范围内

    # 过滤只保留在图像内部的点
    x, y = x[inside], y[inside]

    # 计算框的最小值和最大值，形成 (x1, y1, x2, y2) 格式
    return np.array([x.min(), y.min(), x.max(), y.max()]) if any(x) else np.zeros((1, 4))  # 如果没有有效点，返回全零的框


def segments2boxes(segments):
    # 将分段标签转换为框标签，即 (cls, xy1, xy2, ...) 转换为 (cls, xywh)
    boxes = []
    for s in segments:
        x, y = s.T  # segment xy
        # 将分段点的 x 和 y 坐标转换为最小边界框
        boxes.append([x.min(), y.min(), x.max(), y.max()])  # (x1, y1, x2, y2)
    # 将边界框从 (x1, y1, x2, y2) 转换为 (x, y, w, h)
    return xyxy2xywh(np.array(boxes))  # (cls, xywh)


def resample_segments(segments, n=1000):
    # 对每个 (n,2) 的段进行上采样
    for i, s in enumerate(segments):
        x = np.linspace(0, len(s) - 1, n)  # 创建均匀分布的 x 值用于插值
        xp = np.arange(len(s))  # 原始 x 值
        segments[i] = np.concatenate([np.interp(x, xp, s[:, i]) for i in range(2)]) \
            .reshape(2, -1).T  # 将插值结果重塑为 (2, -1) 的数组，并转置
    return segments


def scale_coords(img1_shape, coords, img0_shape, ratio_pad=None):
    # 将坐标 (xyxy) 从 img1_shape 重新缩放到 img0_shape
    if ratio_pad is None:  # 如果没有提供 ratio_pad，从 img0_shape 计算
        gain = min(img1_shape[0] / img0_shape[0], img1_shape[1] / img0_shape[1])  # 计算缩放因子
        pad = (img1_shape[1] - img0_shape[1] * gain) / 2, (img1_shape[0] - img0_shape[0] * gain) / 2  # 计算填充
    else:
        gain = ratio_pad[0][0]
        pad = ratio_pad[1]

    coords[:, [0, 2]] -= pad[0]  # 应用 x 轴填充
    coords[:, [1, 3]] -= pad[1]  # 应用 y 轴填充
    coords[:, :4] /= gain  # 反向应用缩放因子
    clip_coords(coords, img0_shape)  # 限制坐标在图像边界内
    return coords



def clip_coords(boxes, shape):
    # 将边界框 (xyxy) 限制在图像尺寸 (height, width) 内
    if isinstance(boxes, torch.Tensor):  # 对于单独的 Tensor，更快
        boxes[:, 0].clamp_(0, shape[1])  # 限制 x1
        boxes[:, 1].clamp_(0, shape[0])  # 限制 y1
        boxes[:, 2].clamp_(0, shape[1])  # 限制 x2
        boxes[:, 3].clamp_(0, shape[0])  # 限制 y2
    else:  # 对于 np.array，更快
        boxes[:, [0, 2]] = boxes[:, [0, 2]].clip(0, shape[1])  # 限制 x1 和 x2
        boxes[:, [1, 3]] = boxes[:, [1, 3]].clip(0, shape[0])  # 限制 y1 和 y2


def non_max_suppression(prediction, conf_thres=0.25, iou_thres=0.45, classes=None, agnostic=False, multi_label=False,
                        labels=(), max_det=300):
    """对推理结果进行非极大值抑制 (NMS)

    返回:
         每张图像的检测结果列表，每个图像是一个 (n,6) 的张量 [xyxy, conf, cls]
    """

    nc = prediction.shape[2] - 5  # 类别数量
    xc = prediction[..., 4] > conf_thres  # 置信度候选

    # 检查
    assert 0 <= conf_thres <= 1, f'无效的置信度阈值 {conf_thres}，有效值范围是 0.0 到 1.0'
    assert 0 <= iou_thres <= 1, f'无效的 IoU 阈值 {iou_thres}，有效值范围是 0.0 到 1.0'

    # 设置
    min_wh, max_wh = 2, 4096  # (像素) 最小和最大框宽高
    max_nms = 30000  # 传递到 torchvision.ops.nms() 的最大框数量
    time_limit = 10.0  # 超过此时间后退出
    redundant = True  # 需要冗余检测
    multi_label &= nc > 1  # 每个框多个标签 (增加 0.5ms/图像)
    merge = False  # 使用合并 NMS

    t = time.time()
    output = [torch.zeros((0, 6), device=prediction.device)] * prediction.shape[0]
    for xi, x in enumerate(prediction):  # 图像索引，图像推理
        # 应用约束
        # x[((x[..., 2:4] < min_wh) | (x[..., 2:4] > max_wh)).any(1), 4] = 0  # 宽高
        x = x[xc[xi]]  # 置信度

        # 如果有自动标记的标签则拼接
        if labels and len(labels[xi]):
            l = labels[xi]
            v = torch.zeros((len(l), nc + 5), device=x.device)
            v[:, :4] = l[:, 1:5]  # 框
            v[:, 4] = 1.0  # 置信度
            v[range(len(l)), l[:, 0].long() + 5] = 1.0  # 类别
            x = torch.cat((x, v), 0)

        # 如果没有剩余的框，处理下一张图像
        if not x.shape[0]:
            continue

        # 计算置信度
        x[:, 5:] *= x[:, 4:5]  # 置信度 = 对象置信度 * 类别置信度

        # 框 (中心 x, 中心 y, 宽, 高) 转换为 (x1, y1, x2, y2)
        box = xywh2xyxy(x[:, :4])

        # 检测矩阵 nx6 (xyxy, 置信度, 类别)
        if multi_label:
            i, j = (x[:, 5:] > conf_thres).nonzero(as_tuple=False).T
            x = torch.cat((box[i], x[i, j + 5, None], j[:, None].float()), 1)
        else:  # 仅最佳类别
            conf, j = x[:, 5:].max(1, keepdim=True)
            x = torch.cat((box, conf, j.float()), 1)[conf.view(-1) > conf_thres]

        # 按类别筛选
        if classes is not None:
            x = x[(x[:, 5:6] == torch.tensor(classes, device=x.device)).any(1)]

        # 应用有限约束
        # if not torch.isfinite(x).all():
        #     x = x[torch.isfinite(x).all(1)]

        # 检查形状
        n = x.shape[0]  # 框数量
        if not n:  # 没有框
            continue
        elif n > max_nms:  # 框过多
            x = x[x[:, 4].argsort(descending=True)[:max_nms]]  # 按置信度排序

        # 批量 NMS
        c = x[:, 5:6] * (0 if agnostic else max_wh)  # 类别
        boxes, scores = x[:, :4] + c, x[:, 4]  # 框 (按类别偏移), 置信度
        i = torchvision.ops.nms(boxes, scores, iou_thres)  # NMS
        if i.shape[0] > max_det:  # 限制检测数量
            i = i[:max_det]
        if merge and (1 < n < 3E3):  # 合并 NMS (框使用加权平均合并)
            # 更新框作为 boxes(i,4) = weights(i,n) * boxes(n,4)
            iou = box_iou(boxes[i], boxes) > iou_thres  # iou 矩阵
            weights = iou * scores[None]  # 框权重
            x[i, :4] = torch.mm(weights, x[:, :4]).float() / weights.sum(1, keepdim=True)  # 合并的框
            if redundant:
                i = i[iou.sum(1) > 1]  # 需要冗余

        output[xi] = x[i]
        if (time.time() - t) > time_limit:
            print(f'警告: 超过 NMS 时间限制 {time_limit}s')
            break  # 超过时间限制
    return output



def strip_optimizer(f='best.pt', s=''):  # from utils.general import *; strip_optimizer()
    # 从 'f' 中去除优化器信息，以完成训练，结果可选择性保存为 's'
    x = torch.load(f, map_location=torch.device('cpu'))
    if x.get('ema'):
        x['model'] = x['ema']  # 用 ema 替换模型
    for k in 'optimizer', 'training_results', 'wandb_id', 'ema', 'updates':  # 去除这些键
        x[k] = None
    x['epoch'] = -1
    x['model'].half()  # 转为 FP16
    for p in x['model'].parameters():
        p.requires_grad = False
    torch.save(x, s or f)
    mb = os.path.getsize(s or f) / 1E6  # 文件大小
    print(f"Optimizer stripped from {f},{(' saved as %s,' % s) if s else ''} {mb:.1f}MB")

def print_mutation(results, hyp, save_dir, bucket):
    evolve_csv, results_csv, evolve_yaml = save_dir / 'evolve.csv', save_dir / 'results.csv', save_dir / 'hyp_evolve.yaml'
    keys = ('metrics/precision', 'metrics/recall', 'metrics/mAP_0.5', 'metrics/mAP_0.5:0.95',
            'val/box_loss', 'val/obj_loss', 'val/cls_loss') + tuple(hyp.keys())  # [results + hyps]
    keys = tuple(x.strip() for x in keys)
    vals = results + tuple(hyp.values())
    n = len(keys)

    # 下载（可选）
    if bucket:
        url = f'gs://{bucket}/evolve.csv'
        if gsutil_getsize(url) > (os.path.getsize(evolve_csv) if os.path.exists(evolve_csv) else 0):
            os.system(f'gsutil cp {url} {save_dir}')  # 如果 evolve.csv 大于本地文件，则下载

    # 记录到 evolve.csv
    s = '' if evolve_csv.exists() else (('%20s,' * n % keys).rstrip(',') + '\n')  # 添加表头
    with open(evolve_csv, 'a') as f:
        f.write(s + ('%20.5g,' * n % vals).rstrip(',') + '\n')

    # 打印到屏幕
    print(colorstr('evolve: ') + ', '.join(f'{x.strip():>20s}' for x in keys))
    print(colorstr('evolve: ') + ', '.join(f'{x:20.5g}' for x in vals), end='\n\n\n')

    # 保存为 yaml
    with open(evolve_yaml, 'w') as f:
        data = pd.read_csv(evolve_csv)
        data = data.rename(columns=lambda x: x.strip())  # 去除键的多余空格
        i = np.argmax(fitness(data.values[:, :7]))  # 计算最佳适应度
        f.write('# YOLOv3 Hyperparameter Evolution Results\n' +
                f'# Best generation: {i}\n' +
                f'# Last generation: {len(data)}\n' +
                '# ' + ', '.join(f'{x.strip():>20s}' for x in keys[:7]) + '\n' +
                '# ' + ', '.join(f'{x:>20.5g}' for x in data.values[i, :7]) + '\n\n')
        yaml.safe_dump(hyp, f, sort_keys=False)

    if bucket:
        os.system(f'gsutil cp {evolve_csv} {evolve_yaml} gs://{bucket}')  # 上传到指定 bucket


def apply_classifier(x, model, img, im0):
    # 对 YOLO 输出应用第二阶段分类器
    # 示例模型 = torchvision.models.__dict__['efficientnet_b0'](pretrained=True).to(device).eval()
    im0 = [im0] if isinstance(im0, np.ndarray) else im0
    for i, d in enumerate(x):  # 针对每张图片
        if d is not None and len(d):
            d = d.clone()

            # 重新调整和填充切割区域
            b = xyxy2xywh(d[:, :4])  # 转换为 [x, y, w, h]
            b[:, 2:] = b[:, 2:].max(1)[0].unsqueeze(1)  # 转换为正方形
            b[:, 2:] = b[:, 2:] * 1.3 + 30  # 填充
            d[:, :4] = xywh2xyxy(b).long()  # 转换回 [x1, y1, x2, y2]

            # 将框的坐标从 img_size 调整到 im0 大小
            scale_coords(img.shape[2:], d[:, :4], im0[i].shape)

            # 类别预测
            pred_cls1 = d[:, 5].long()  # 原始类别预测
            ims = []
            for j, a in enumerate(d):  # 针对每个检测框
                cutout = im0[i][int(a[1]):int(a[3]), int(a[0]):int(a[2])]  # 裁剪图像区域
                im = cv2.resize(cutout, (224, 224))  # 调整大小到 224x224 BGR
                # cv2.imwrite('example%i.jpg' % j, cutout)

                im = im[:, :, ::-1].transpose(2, 0, 1)  # BGR 转 RGB，调整为 3x224x224
                im = np.ascontiguousarray(im, dtype=np.float32)  # uint8 转 float32
                im /= 255  # 0 - 255 转 0.0 - 1.0
                ims.append(im)

            pred_cls2 = model(torch.Tensor(ims).to(d.device)).argmax(1)  # 分类器预测
            x[i] = x[i][pred_cls1 == pred_cls2]  # 保留匹配的分类检测

    return x


def increment_path(path, exist_ok=False, sep='', mkdir=False):
    # 增加文件或目录路径，例如 runs/exp --> runs/exp{sep}2, runs/exp{sep}3, ... 等等
    path = Path(path)  # 兼容操作系统
    if path.exists() and not exist_ok:
        path, suffix = (path.with_suffix(''), path.suffix) if path.is_file() else (path, '')
        dirs = glob.glob(f"{path}{sep}*")  # 类似路径
        matches = [re.search(rf"%s{sep}(\d+)" % path.stem, d) for d in dirs]
        i = [int(m.groups()[0]) for m in matches if m]  # 索引
        n = max(i) + 1 if i else 2  # 增量数
        path = Path(f"{path}{sep}{n}{suffix}")  # 增加路径
    if mkdir:
        path.mkdir(parents=True, exist_ok=True)  # 创建目录
    return path

# Variables
NCOLS = 0 if is_docker() else shutil.get_terminal_size().columns  # 终端窗口大小