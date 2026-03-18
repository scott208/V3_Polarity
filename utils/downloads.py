# YOLOv3 🚀 by Ultralytics, GPL-3.0 license
"""
Download utils
"""

import os
import platform
import subprocess
import time
import urllib
from pathlib import Path
from zipfile import ZipFile

import requests
import torch


def gsutil_getsize(url=''):
    # 获取 gs://bucket/file 的大小 https://cloud.google.com/storage/docs/gsutil/commands/du
    s = subprocess.check_output(f'gsutil du {url}', shell=True).decode('utf-8')
    return eval(s.split(' ')[0]) if len(s) else 0  # 字节

def safe_download(file, url, url2=None, min_bytes=1E0, error_msg=''):
    # 尝试从 url 或 url2 下载文件，检查并删除小于 min_bytes 的不完整下载文件
    file = Path(file)
    assert_msg = f"Downloaded file '{file}' does not exist or size is < min_bytes={min_bytes}"
    try:  # 尝试使用 url 下载文件
        print(f'Downloading {url} to {file}...')
        torch.hub.download_url_to_file(url, str(file))
        assert file.exists() and file.stat().st_size > min_bytes, assert_msg  # 检查文件是否存在且大小足够
    except Exception as e:  # 如果失败，则使用 url2 重新尝试下载
        file.unlink(missing_ok=True)  # 删除部分下载的文件
        print(f'ERROR: {e}\nRe-attempting {url2 or url} to {file}...')
        os.system(f"curl -L '{url2 or url}' -o '{file}' --retry 3 -C -")  # 使用 curl 下载，支持重试和断点续传
    finally:
        if not file.exists() or file.stat().st_size < min_bytes:  # 检查文件是否存在且大小足够
            file.unlink(missing_ok=True)  # 删除部分下载的文件
            print(f"ERROR: {assert_msg}\n{error_msg}")
        print('')

def attempt_download(file, repo='ultralytics/yolov3'):  # 从 utils.downloads 导入 *; attempt_download()
    # 如果文件不存在，尝试下载
    file = Path(str(file).strip().replace("'", ''))

    if not file.exists():
        # URL 指定
        name = Path(urllib.parse.unquote(str(file))).name  # 解码 '%2F' 为 '/' 等
        if str(file).startswith(('http:/', 'https:/')):  # 下载
            url = str(file).replace(':/', '://')  # Pathlib 将 :// 转为 :/
            name = name.split('?')[0]  # 解析认证 https://url.com/file.txt?auth...
            safe_download(file=name, url=url, min_bytes=1E5)
            return name

        # GitHub 资产
        file.parent.mkdir(parents=True, exist_ok=True)  # 创建父目录（如有必要）
        try:
            response = requests.get(f'https://api.github.com/repos/{repo}/releases/latest').json()  # GitHub API
            assets = [x['name'] for x in response['assets']]  # 版本资产，例如 ['yolov3.pt'...]
            tag = response['tag_name']  # 例如 'v1.0'
        except:  # 回退计划
            assets = ['yolov3.pt', 'yolov3-spp.pt', 'yolov3-tiny.pt']
            try:
                tag = subprocess.check_output('git tag', shell=True, stderr=subprocess.STDOUT).decode().split()[-1]
            except:
                tag = 'v9.5.0'  # 当前版本

        if name in assets:
            safe_download(file,
                          url=f'https://github.com/{repo}/releases/download/{tag}/{name}',
                          # url2=f'https://storage.googleapis.com/{repo}/ckpt/{name}',  # 备份 URL（可选）
                          min_bytes=1E5,
                          error_msg=f'{file} missing, try downloading from https://github.com/{repo}/releases/')
    return str(file)


def gdrive_download(id='16TiPfZj7htmTyhntwcZyEEAejOUxuT6m', file='tmp.zip'):
    # 从Google Drive下载文件。用法示例：from yolov3.utils.downloads import *; gdrive_download()
    t = time.time()  # 记录开始时间
    file = Path(file)  # 将传入的文件名转换为Path对象
    cookie = Path('cookie')  # 存储gdrive cookie的文件路径
    print(f'Downloading https://drive.google.com/uc?export=download&id={id} as {file}... ', end='')
    file.unlink(missing_ok=True)  # 如果文件已存在，则删除它
    cookie.unlink(missing_ok=True)  # 如果cookie文件已存在，则删除它

    # 尝试下载文件
    out = "NUL" if platform.system() == "Windows" else "/dev/null"  # 根据操作系统选择空输出设备
    os.system(f'curl -c ./cookie -s -L "drive.google.com/uc?export=download&id={id}" > {out}')  # 执行curl命令以获取cookie
    if os.path.exists('cookie'):  # 如果存在cookie，说明需要处理大文件
        s = f'curl -Lb ./cookie "drive.google.com/uc?export=download&confirm={get_token()}&id={id}" -o {file}'  # 使用确认token下载大文件
    else:  # 否则处理小文件
        s = f'curl -s -L -o {file} "drive.google.com/uc?export=download&id={id}"'  # 直接下载小文件
    r = os.system(s)  # 执行curl命令，并捕获返回值
    cookie.unlink(missing_ok=True)  # 删除cookie文件

    # 错误检查
    if r != 0:  # 如果下载命令返回值不为0，则表示发生错误
        file.unlink(missing_ok=True)  # 删除部分下载的文件
        print('Download error ')  # 打印下载错误信息
        return r  # 返回错误码

    # 如果文件是压缩文件，则解压缩
    if file.suffix == '.zip':  # 判断文件后缀是否为.zip
        print('unzipping... ', end='')  # 打印解压信息
        ZipFile(file).extractall(path=file.parent)  # 解压缩文件到文件所在目录
        file.unlink()  # 删除zip文件

    print(f'Done ({time.time() - t:.1f}s)')  # 打印完成信息以及下载耗时
    return r  # 返回命令执行的结果


def get_token(cookie="./cookie"):
    # 从指定的cookie文件中提取下载确认token
    with open(cookie) as f:  # 打开cookie文件
        for line in f:  # 遍历文件中的每一行
            if "download" in line:  # 查找包含"download"的行
                return line.split()[-1]  # 返回该行最后一个空格分隔的字段作为token
    return ""  # 如果没有找到符合条件的行，则返回空字符串

# Google utils: https://cloud.google.com/storage/docs/reference/libraries ----------------------------------------------
#
#
# def upload_blob(bucket_name, source_file_name, destination_blob_name):
#     # Uploads a file to a bucket
#     # https://cloud.google.com/storage/docs/uploading-objects#storage-upload-object-python
#
#     storage_client = storage.Client()
#     bucket = storage_client.get_bucket(bucket_name)
#     blob = bucket.blob(destination_blob_name)
#
#     blob.upload_from_filename(source_file_name)
#
#     print('File {} uploaded to {}.'.format(
#         source_file_name,
#         destination_blob_name))
#
#
# def download_blob(bucket_name, source_blob_name, destination_file_name):
#     # Uploads a blob from a bucket
#     storage_client = storage.Client()
#     bucket = storage_client.get_bucket(bucket_name)
#     blob = bucket.blob(source_blob_name)
#
#     blob.download_to_filename(destination_file_name)
#
#     print('Blob {} downloaded to {}.'.format(
#         source_blob_name,
#         destination_file_name))
