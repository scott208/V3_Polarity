# YOLOv3 🚀 by Ultralytics, GPL-3.0 license
"""
在图像、视频、目录、流等上运行推理。

用法:
    $ python path/to/detect.py --weights yolov3.pt --source 0  # 摄像头
                                                             img.jpg  # 图像
                                                             vid.mp4  # 视频
                                                             path/  # 目录
                                                             path/*.jpg  # 通配符
                                                             'https://youtu.be/Zgi9g1ksQHc'  # YouTube
                                                             'rtsp://example.com/media.mp4'  # RTSP, RTMP, HTTP 流
"""

import argparse
import os
import sys
from pathlib import Path

import cv2
import torch
import torch.backends.cudnn as cudnn

# 获取当前文件的绝对路径
FILE = Path(__file__).resolve()
# 获取当前文件所在的根目录
ROOT = FILE.parents[0]  # 根目录
# 如果根目录不在系统路径中，则添加根目录到系统路径
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # 添加根目录到系统路径
# 将根目录相对化
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # 相对路径

# 导入自定义模块
from models.common import DetectMultiBackend
from utils.datasets import IMG_FORMATS, VID_FORMATS, LoadImages, LoadStreams
from utils.general import (LOGGER, check_file, check_img_size, check_imshow, check_requirements, colorstr,
                           increment_path, non_max_suppression, print_args, scale_coords, strip_optimizer, xyxy2xywh)
from utils.plots import Annotator, colors, save_one_box
from utils.torch_utils import select_device, time_sync



@torch.no_grad()  # @torch.no_grad() 是一个 PyTorch 的装饰器，作用是禁用梯度计算。它通常用于推理（inference）阶段，以节省内存并加快计算速度。

def run(weights=ROOT / 'yolov3.pt',  # 模型路径
        source=ROOT / 'data/images',  # 文件/目录/URL/通配符，0 表示摄像头
        imgsz=640,  # 推理图像大小（像素）
        conf_thres=0.25,  # 置信度阈值
        iou_thres=0.45,  # 非极大值抑制（NMS）IOU 阈值
        max_det=1000,  # 每张图像的最大检测数
        device='',  # CUDA 设备，例如 0 或 0,1,2,3 或 CPU
        view_img=False,  # 显示结果
        save_txt=False,  # 将结果保存到 *.txt
        save_conf=False,  # 在保存的标签中包含置信度
        save_crop=False,  # 保存裁剪后的预测框
        nosave=False,  # 不保存图像/视频
        classes=None,  # 按类别过滤：--class 0 或 --class 0 2 3
        agnostic_nms=False,  # 类别无关的 NMS
        augment=False,  # 增强推理
        visualize=False,  # 可视化特征
        update=False,  # 更新所有模型
        project=ROOT / 'runs/detect',  # 保存结果的项目路径
        name='exp',  # 保存结果的项目名称
        exist_ok=False,  # 允许现有的项目名称，不递增
        line_thickness=3,  # 边界框厚度（像素）
        hide_labels=False,  # 隐藏标签
        hide_conf=False,  # 隐藏置信度
        half=False,  # 使用 FP16 半精度推理
        dnn=False,  # 使用 OpenCV DNN 进行 ONNX 推理
        ):
    # ===================================== 1、初始化一些配置 =====================================
    # 是否保存预测后的图片 默认nosave=False 所以只要传入的文件地址不是以.txt结尾 就都是要保存预测后的图片的
    source = str(source)
    save_img = not nosave and not source.endswith('.txt')  # 是否保存推理图像
    is_file = Path(source).suffix[1:] in (IMG_FORMATS + VID_FORMATS)  # 检查是否为文件
    is_url = source.lower().startswith(('rtsp://', 'rtmp://', 'http://', 'https://'))  # 检查是否为 URL
    webcam = source.isnumeric() or source.endswith('.txt') or (is_url and not is_file)  # 检查是否为摄像头输入
    if is_url and is_file:
        source = check_file(source)  # 下载文件

    # 创建保存目录
    save_dir = increment_path(Path(project) / name, exist_ok=exist_ok)  # 递增运行目录
    (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # 创建目录

    # ===================================== 2、载入模型和模型参数并调整模型 =====================================
    # 加载模型
    device = select_device(device)  # 选择设备
    model = DetectMultiBackend(weights, device=device, dnn=dnn)  # 加载模型
    stride, names, pt, jit, onnx = model.stride, model.names, model.pt, model.jit, model.onnx # 获取模型属性
    imgsz = check_img_size(imgsz, s=stride)  # 检查图像大小

    # 设置半精度,# 如果设配是GPU 就使用half(float16)  包括模型半精度和输入图片半精度
    half &= pt and device.type != 'cpu'  # 半精度仅支持在 CUDA 上的 PyTorch
    if pt:
        model.model.half() if half else model.model.float()

    # ===================================== 3、加载推理数据 =====================================
    # Set Dataloader
    # 通过不同的输入源来设置不同的数据加载方式
    # Dataloader
    if webcam:
        view_img = check_imshow()  # 检查是否可以显示图像
        cudnn.benchmark = True  # 设置为 True 可以加速恒定图像大小的推理
        dataset = LoadStreams(source, img_size=imgsz, stride=stride, auto=pt and not jit)  # 加载流数据
        bs = len(dataset)  # 批量大小
    else:
        # 一般是直接从source文件目录下直接读取图片或者视频数据
        dataset = LoadImages(source, img_size=imgsz, stride=stride, auto=pt and not jit)  # 加载图像数据
        bs = 1  # batch_size
    vid_path, vid_writer = [None] * bs, [None] * bs  # 通过不同的输入源来设置不同的数据加载方式

    # 如果使用 PyTorch（pt=True）且设备类型不是 'cpu'：
    if pt and device.type != 'cpu':
        # 运行模型的预热步骤，使用全零张量作为输入，并将其移动到指定的设备，并确保数据类型与模型参数的数据类型匹配
        model(torch.zeros(1, 3, *imgsz).to(device).type_as(next(model.model.parameters())))  # warmup
    dt, seen = [0.0, 0.0, 0.0], 0  # 初始化变量 dt 和 seen

    # ===================================== 5、正式推理 =====================================
    for path, im, im0s, vid_cap, s in dataset:
        # path: 图片/视频的路径
        # img: 进行resize + pad之后的图片
        # img0s: 原尺寸的图片
        # vid_cap: 当读取图片时为None, 读取视频时为视频源

        # 5.1、对每张图片 / 视频进行前向推理
        t1 = time_sync()

        im = torch.from_numpy(im).to(device)   # 5.2、处理每一张图片/视频的格式
        im = im.half() if half else im.float()  # 半精度训练 uint8 to fp16/32
        im /= 255  # 归一化 0 - 255 to 0.0 - 1.0
        # 如果图片是3维(RGB) 就在前面添加一个维度1当中batch_size=1
        # 因为输入网络的图片需要是4为的 [batch_size, channel, w, h]
        if len(im.shape) == 3:
            im = im[None]  # expand for batch dim
        t2 = time_sync()  # 获取当前时间并进行时间同步
        dt[0] += t2 - t1  # 累加时间差到 dt[0] 中

        # 如果需要可视化，则设置保存路径并创建目录
        visualize = increment_path(save_dir / Path(path).stem, mkdir=True) if visualize else False
        # 使用模型进行预测，同时根据需要进行数据增强和可视化
        pred = model(im, augment=augment, visualize=visualize)
        # 获取当前时间并累加到 dt[1] 中
        t3 = time_sync()
        dt[1] += t3 - t2

        # NMS
        # Apply NMS  进行NMS
        # conf_thres: 置信度阈值
        # iou_thres: iou阈值
        # classes: 是否只保留特定的类别 默认为None
        # agnostic_nms: 进行nms是否也去除不同类别之间的框 默认False
        # max_det: 每张图片的最大目标个数 默认1000
        # pred: [num_obj, 6] = [5, 6]   这里的预测信息pred还是相对于 img_size(640) 的
        pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)
        dt[2] += time_sync() - t3

        # Second-stage classifier (optional)
        # pred = utils.general.apply_classifier(pred, classifier_model, im, im0s)

        # 后续保存或者打印预测信息
        # 对每张图片进行处理  将pred(相对img_size 640)映射回原图img0 size
        for i, det in enumerate(pred):  # per image
            seen += 1
            if webcam:  # batch_size >= 1
                # 如果输入源是webcam（网页）则batch_size>=1 取出dataset中的一张图片
                p, im0, frame = path[i], im0s[i].copy(), dataset.count
                s += f'{i}: '
            else:
                # 但是大部分我们一般都是从LoadImages流读取本都文件中的照片或者视频 所以batch_size=1
                # p: 当前图片/视频的绝对路径
                # s: 输出信息 初始为 ''
                # im0: 原始图片 letterbox + pad 之前的图片
                # frame: 初始为0  可能是当前图片属于视频中的第几帧？
                p, im0, frame = path, im0s.copy(), getattr(dataset, 'frame', 0)

            # 当前图片路径
            p = Path(p)  # to Path
            # 图片/视频的保存路径save_path
            save_path = str(save_dir / p.name)  # im.jpg
            # txt文件(保存预测框坐标)保存路径
            txt_path = str(save_dir / 'labels' / p.stem) + ('' if dataset.mode == 'image' else f'_{frame}')  # im.txt
            s += '%gx%g ' % im.shape[2:]  # print string  输出信息  图片shape (w, h)
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh  gain gn = [w, h, w, h]  用于后面的归一化
            imc = im0.copy() if save_crop else im0  # imc: for save_crop 在save_crop中使用
            annotator = Annotator(im0, line_width=line_thickness, example=str(names))


            if len(det):
                # Rescale boxes from img_size to im0 size
                # 将预测信息（相对img_size 640）映射回原图 img0 size
                det[:, :4] = scale_coords(im.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                # 输出信息s + 检测到的各个类别的目标个数
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

                # Write results
                # 保存预测信息: txt、img0上画框、crop_img
                for *xyxy, conf, cls in reversed(det):
                    # 将每个图片的预测信息分别存入save_dir/labels下的xxx.txt中 每行: class_id+score+xywh
                    if save_txt:  # Write to file
                        # 将xyxy(左上角 + 右下角)格式转换为xywh(中心的 + 宽高)格式 并除以gn(whwh)做归一化 转为list再保存
                        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                        line = (cls, *xywh, conf) if save_conf else (cls, *xywh)  # label format
                        with open(txt_path + '.txt', 'a') as f:
                            f.write(('%g ' * len(line)).rstrip() % line + '\n')

                    # 在原图上画框 + 将预测到的目标剪切出来 保存成图片 保存在save_dir/crops下
                    if save_img or save_crop or view_img:  # Add bbox to image
                        c = int(cls)  # integer class
                        label = None if hide_labels else (names[c] if hide_conf else f'{names[c]} {conf:.2f}')
                        annotator.box_label(xyxy, label, color=colors(c, True))
                        if save_crop:
                            # 如果需要就将预测到的目标剪切出来保存成图片 保存在save_dir/crops下
                            save_one_box(xyxy, imc, file=save_dir / 'crops' / names[c] / f'{p.stem}.jpg', BGR=True)

            # Print time (inference-only)
            # LOGGER.info(f'{s}Done. ({t3 - t2:.3f}s)')
            print(f'{s}Done. ({t3 - t2:.3f}s)')

            # 是否需要显示我们预测后的结果  img0(此时已将pred结果可视化到了img0中)
            im0 = annotator.result()
            if view_img:
                cv2.imshow(str(p), im0)
                cv2.waitKey(1)  # 1 millisecond

            # 是否需要保存图片或视频（检测后的图片/视频 里面已经被我们画好了框的） img0
            if save_img:  # 如果需要保存图像
                if dataset.mode == 'image':  # 如果数据集模式为 'image'
                    cv2.imwrite(save_path, im0)  # 保存单张图像
                else:  # 数据集模式为 'video' 或 'stream'
                    if vid_path[i] != save_path:  # 如果保存路径与当前视频路径不同（表示新视频）
                        vid_path[i] = save_path
                        if isinstance(vid_writer[i], cv2.VideoWriter):
                            vid_writer[i].release()  # 释放之前的视频写入器
                        if vid_cap:  # 如果是视频文件
                            fps = vid_cap.get(cv2.CAP_PROP_FPS)  # 获取帧率
                            w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))  # 获取帧宽度
                            h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))  # 获取帧高度
                        else:  # 如果是实时流
                            fps, w, h = 30, im0.shape[1], im0.shape[0]  # 设置默认帧率和图像尺寸
                            save_path += '.mp4'  # 添加文件扩展名
                        # 创建新的视频写入器
                        vid_writer[i] = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
                    # 将当前帧写入视频
                    vid_writer[i].write(im0)
    # Print results
    t = tuple(x / seen * 1E3 for x in dt)  # 计算每张图像的处理速度（毫秒）
    # LOGGER.info(f'Speed: %.1fms pre-process, %.1fms inference, %.1fms NMS per image at shape {(1, 3, *imgsz)}' % t)
    print(f'Speed: %.1fms pre-process, %.1fms inference, %.1fms NMS per image at shape {(1, 3, *imgsz)}' % t)
    if save_txt or save_img:  # 如果需要保存文本或图像结果
        # 检查是否保存了文本标签，并打印保存信息
        s = (f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}"
             if save_txt else '')
        LOGGER.info(f"Results saved to {colorstr('bold', save_dir)}{s}")
    if update:  # 如果需要更新模型
        strip_optimizer(weights)  # 去除优化器（strip_optimizer函数用于修复 SourceChangeWarning 问题）

def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default=ROOT / 'weight/yolov3.pt', help='model path(s)')  # weights: 模型的权重地址 默认 weights/best.pt
    parser.add_argument('--source', type=str, default=ROOT / 'data/images', help='file/dir/URL/glob, 0 for webcam')  # source: 测试数据文件(图片或视频)的保存路径 默认data/images
    parser.add_argument('--imgsz', '--img', '--img-size', nargs='+', type=int, default=[416], help='inference size h,w')  # imgsz: 网络输入图片的大小 默认640
    parser.add_argument('--conf-thres', type=float, default=0.6, help='confidence threshold') # conf-thres: object置信度阈值 默认0.25
    parser.add_argument('--iou-thres', type=float, default=0.5, help='NMS IoU threshold')  # iou-thres: 做nms的iou阈值 默认0.45
    parser.add_argument('--max-det', type=int, default=1000, help='maximum detections per image')   # max-det: 每张图片最大的目标个数 默认1000
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')  # device: 设置代码执行的设备 cuda device, i.e. 0 or 0,1,2,3 or cpu
    parser.add_argument('--view-img', action='store_true', default=True, help='show results')  # view-img: 是否展示预测之后的图片或视频 默认False
    parser.add_argument('--save-txt', action='store_true', default=True, help='save results to *.txt')  # save-txt: 是否将预测的框坐标以txt文件格式保存 默认False 会在runs/detect/expn/labels下生成每张图片预测的txt文件
    parser.add_argument('--save-conf', action='store_true', default=True, help='save confidences in --save-txt labels')  # save-conf: 是否保存预测每个目标的置信度到预测tx文件中 默认False
    parser.add_argument('--save-crop', action='store_true', default=True, help='save cropped prediction boxes')  # save-crop: 是否需要将预测到的目标从原图中扣出来 剪切好 并保存 会在runs/detect/expn下生成crops文件，将剪切的图片保存在里面  默认False
    parser.add_argument('--nosave', action='store_true', help='do not save images/vidruns/train/exp/weights/best.pteos')  # nosave: 是否不要保存预测后的图片  默认False 就是默认要保存预测后的图片
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --classes 0, or --classes 0 2 3')  # classes: 在nms中是否是只保留某些特定的类 默认是None 就是所有类只要满足条件都可以保留, default=[0,6,1,8,9, 7]
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')  # agnostic-nms: 进行nms是否也除去不同类别之间的框 默认False
    parser.add_argument('--augment', action='store_true', help='augmented inference')  # 是否使用数据增强进行推理，默认为False
    parser.add_argument('--visualize', action='store_true', help='visualize features')  #  -visualize:是否可视化特征图，默认为 False
    parser.add_argument('--update', action='store_true', help='update all models')  # -update: 如果为True，则对所有模型进行strip_optimizer操作，去除pt文件中的优化器等信息，默认为False
    parser.add_argument('--project', default=ROOT / 'runs/detect', help='save results to project/name')  # project: 当前测试结果放在哪个主文件夹下 默认runs/detect
    parser.add_argument('--name', default='exp', help='save results to project/name')  # name: 当前测试结果放在run/detect下的文件名  默认是exp
    parser.add_argument('--exist-ok', action='store_true', default=False, help='existing project/name ok, do not increment')  # -exist-ok: 是否覆盖已有结果，默认为 False
    parser.add_argument('--line-thickness', default=3, type=int, help='bounding box thickness (pixels)')  # -line-thickness:画 bounding box 时的线条宽度，默认为 3
    parser.add_argument('--hide-labels', default=False, action='store_true', help='hide labels')  # -hide-labels:是否隐藏标签信息，默认为 False
    parser.add_argument('--hide-conf', default=False, action='store_true', help='hide confidences')  # -hide-conf:是否隐藏置信度信息，默认为 False
    parser.add_argument('--half', action='store_true', help='use FP16 half-precision inference')  # half: 是否使用半精度 Float16 推理 可以缩短推理时间 但是默认是False
    parser.add_argument('--dnn', action='store_true', help='use OpenCV DNN for ONNX inference')  # -dnn:是否使用 OpenCV DNN 进行 ONNX 推理，默认为 False
    opt = parser.parse_args()  # 解析命令行参数，并将结果存储在opt对象中
    opt.imgsz *= 2 if len(opt.imgsz) == 1 else 1  # 如果imgsz参数的长度为1，则将其值乘以2；否则保持不变
    print_args(FILE.stem, opt)  #  打印解析后的参数，FILE.stem是文件的名称（不含扩展名）
    return opt

def main(opt):
    check_requirements(exclude=('tensorboard', 'thop'))  # 检查项目所需的依赖项，排除 'tensorboard' 和 'thop' 这两个库
    run(**vars(opt))  # 使用命令行参数的字典形式调用 run 函数


if __name__ == "__main__":
    # 这是 Python 中的一个惯用语法，
    # 它确保以下的代码块只有在当前脚本作为主程序运行时才会被执行，而不是作为模块被导入时执行。
    opt = parse_opt()
    main(opt)
