# YOLOv3 🚀 by Ultralytics, GPL-3.0 license
"""
PyTorch Hub models https://pytorch.org/hub/ultralytics_yolov5/

Usage:
    import torch
    model = torch.hub.load('ultralytics/yolov3', 'yolov3')
"""

import torch


def _create(name, pretrained=True, channels=3, classes=80, autoshape=True, verbose=True, device=None):
    """创建指定的模型

    参数:
        name (str): 模型的名称，例如 'yolov3'
        pretrained (bool): 是否加载预训练权重
        channels (int): 输入通道数
        classes (int): 模型的类别数
        autoshape (bool): 是否应用 .autoshape() 包装器到模型
        verbose (bool): 是否打印所有信息到屏幕
        device (str, torch.device, None): 用于模型参数的设备

    返回:
         pytorch 模型
    """

    # 导入必要的库
    from pathlib import Path
    from models.experimental import attempt_load
    from models.yolo import Model
    from utils.downloads import attempt_download
    from utils.general import check_requirements, intersect_dicts, set_logging
    from utils.torch_utils import select_device

    file = Path(__file__).resolve()  # 获取当前文件的绝对路径
    check_requirements(exclude=('tensorboard', 'thop', 'opencv-python'))  # 检查并安装必要的依赖
    set_logging(verbose=verbose)  # 根据 verbose 参数设置日志

    save_dir = Path('') if str(name).endswith('.pt') else file.parent  # 确定保存目录
    path = (save_dir / name).with_suffix('.pt')  # 创建模型检查点的路径

    try:
        # 选择要使用的设备（GPU或CPU）
        device = select_device(('0' if torch.cuda.is_available() else 'cpu') if device is None else device)

        if pretrained and channels == 3 and classes == 80:
            # 如果模型是预训练的并且有默认的通道数和类别数，尝试加载它
            model = attempt_load(path, map_location=device)  # 下载/加载 FP32 模型
        else:
            # 否则，从配置文件创建一个新模型
            cfg = list((Path(__file__).parent / 'models').rglob(f'{name}.yaml'))[0]  # model.yaml 路径
            model = Model(cfg, channels, classes)  # 创建模型
            if pretrained:
                # 如果指定了预训练权重，则加载预训练权重
                ckpt = torch.load(attempt_download(path), map_location=device)  # 加载检查点
                csd = ckpt['model'].float().state_dict()  # 从检查点获取 state_dict
                csd = intersect_dicts(csd, model.state_dict(), exclude=['anchors'])  # 交集 state_dict
                model.load_state_dict(csd, strict=False)  # 将 state_dict 加载到模型中
                if len(ckpt['model'].names) == classes:
                    model.names = ckpt['model'].names  # 设置类别名称属性

        if autoshape:
            model = model.autoshape()  # 对模型应用 autoshape 包装器，以适应不同的输入类型

        return model.to(device)  # 将模型移到指定设备

    except Exception as e:
        # 处理异常并提供帮助链接
        help_url = 'https://github.com/ultralytics/yolov5/issues/36'
        s = 'Cache may be out of date, try `force_reload=True`. See %s for help.' % help_url
        raise Exception(s) from e  # 提示带有额外信息的异常


def custom(path='path/to/model.pt', autoshape=True, verbose=True, device=None):
    # 自定义或本地模型
    return _create(path, autoshape=autoshape, verbose=verbose, device=device)

def yolov3(pretrained=True, channels=3, classes=80, autoshape=True, verbose=True, device=None):
    # YOLOv3 模型 https://github.com/ultralytics/yolov3
    return _create('yolov3', pretrained, channels, classes, autoshape, verbose, device)

def yolov3_spp(pretrained=True, channels=3, classes=80, autoshape=True, verbose=True, device=None):
    # YOLOv3-SPP 模型 https://github.com/ultralytics/yolov3
    return _create('yolov3-spp', pretrained, channels, classes, autoshape, verbose, device)

def yolov3_tiny(pretrained=True, channels=3, classes=80, autoshape=True, verbose=True, device=None):
    # YOLOv3-tiny 模型 https://github.com/ultralytics/yolov3
    return _create('yolov3-tiny', pretrained, channels, classes, autoshape, verbose, device)



if __name__ == '__main__':
    # 创建YOLOv3-tiny模型，使用预训练权重
    model = _create(name='yolov3-tiny', pretrained=True, channels=3, classes=80, autoshape=True,
                    verbose=True)  # pretrained
    # model = custom(path='path/to/model.pt')  # 自定义模型

    # 验证推理
    from pathlib import Path
    import cv2
    import numpy as np
    from PIL import Image

    # 定义要测试的图像列表
    imgs = [
        'data/images/zidane.jpg',  # 文件名
        Path('data/images/zidane.jpg'),  # 文件路径
        'https://ultralytics.com/images/zidane.jpg',  # 网络地址
        cv2.imread('data/images/bus.jpg')[:, :, ::-1],  # 使用OpenCV读取并转换为RGB格式
        Image.open('data/images/bus.jpg'),  # 使用PIL打开图像
        np.zeros((320, 640, 3))  # 创建一个空的numpy数组（黑色图像）
    ]
    # 批量推理
    results = model(imgs)  # 进行推理
    results.print()  # 打印结果
    results.save()  # 保存结果
