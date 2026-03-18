import xml.etree.ElementTree as ET  # 导入用于解析XML文件的库
import pickle  # 导入用于序列化和反序列化Python对象的库
import os  # 导入用于操作文件和目录的库
from os import listdir, getcwd  # 从os库中导入列出目录文件和获取当前工作目录的方法
from os.path import join  # 从os.path库中导入用于路径拼接的方法
import random  # 导入用于生成随机数的库
from shutil import copyfile  # 从shutil库中导入用于复制文件的方法

# •人
# •鸟、猫、牛、狗、马、羊
# •飞机、自行车、船、公共汽车、汽车、摩托车、火车
# •瓶子、椅子、餐桌、盆栽、沙发、电视/显示器

# 定义要检测的类别
classes = ['aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair', 'cow', 'diningtable', 'dog',
           'horse', 'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor']

TRAIN_RATIO = 80  # 定义训练集的比例，80表示80%

def clear_hidden_files(path):
    # 获取指定路径下的所有文件和目录
    dir_list = os.listdir(path)

    for i in dir_list:
        # 获取文件或目录的绝对路径
        abspath = os.path.join(os.path.abspath(path), i)

        if os.path.isfile(abspath):
            # 如果是文件并且文件名以“._”开头，则删除该文件
            if i.startswith("._"):
                os.remove(abspath)
        else:
            # 如果是目录，则递归调用自身
            clear_hidden_files(abspath)


def convert(size, box):
    # 计算图像宽度和高度的倒数
    dw = 1. / size[0]
    dh = 1. / size[1]

    # 计算边界框中心的x和y坐标
    x = (box[0] + box[1]) / 2.0
    y = (box[2] + box[3]) / 2.0

    # 计算边界框的宽度和高度
    w = box[1] - box[0]
    h = box[3] - box[2]

    # 将边界框坐标归一化为相对于图像尺寸的比例
    x = x * dw
    w = w * dw
    y = y * dh
    h = h * dh

    return (x, y, w, h)  # 返回归一化后的边界框坐标


def convert_annotation(image_id):
    # 打开对应的VOC XML标注文件，使用UTF-8编码读取
    in_file = open('VOCdevkit/VOC2007/Annotations/%s.xml' % image_id, encoding="utf-8")
    # 打开或创建一个对应的YOLO格式的文本标注文件，使用UTF-8编码写入
    out_file = open('VOCdevkit/VOC2007/YOLOLabels/%s.txt' % image_id, 'w', encoding="utf-8")

    # 使用 xml.etree.ElementTree 解析XML文件，获取根元素
    tree = ET.parse(in_file)
    root = tree.getroot()

    # 获取图像的尺寸信息（宽度和高度）
    size = root.find('size')
    w = int(size.find('width').text)
    h = int(size.find('height').text)

    # 遍历所有的 object 标签
    for obj in root.iter('object'):
        # 获取困难样本标志，如果为1则跳过
        difficult = obj.find('difficult').text
        # 获取类别名称
        cls = obj.find('name').text
        # 如果类别不在定义的 classes 列表中，或者是困难样本，则跳过
        if cls not in classes or int(difficult) == 1:
            continue
        # 获取类别的索引
        cls_id = classes.index(cls)
        # 获取边界框的坐标（xmin, xmax, ymin, ymax）
        xmlbox = obj.find('bndbox')
        b = (float(xmlbox.find('xmin').text), float(xmlbox.find('xmax').text), float(xmlbox.find('ymin').text),
             float(xmlbox.find('ymax').text))
        # 使用 convert 函数将边界框坐标转换为YOLO格式的相对坐标
        bb = convert((w, h), b)
        # 将类别索引和转换后的坐标写入YOLO标注文件
        out_file.write(str(cls_id) + " " + " ".join([str(a) for a in bb]) + '\n')

    # 关闭输入和输出文件
    in_file.close()
    out_file.close()

# 获取当前工作目录
wd = os.getcwd()

# 构建 "VOCdevkit/" 目录路径
data_base_dir = os.path.join(wd, "VOCdevkit/")
# 如果 "VOCdevkit/" 目录不存在，则创建该目录
if not os.path.isdir(data_base_dir):
    os.mkdir(data_base_dir)

# 构建 "VOCdevkit/VOC2007/" 目录路径
work_sapce_dir = os.path.join(data_base_dir, "VOC2007/")
# 如果 "VOC2007/" 目录不存在，则创建该目录
if not os.path.isdir(work_sapce_dir):
    os.mkdir(work_sapce_dir)

# 构建 "VOCdevkit/VOC2007/Annotations/" 目录路径
annotation_dir = os.path.join(work_sapce_dir, "Annotations/")
# 如果 "Annotations/" 目录不存在，则创建该目录
if not os.path.isdir(annotation_dir):
    os.mkdir(annotation_dir)
# 清除 "Annotations/" 目录中的隐藏文件
clear_hidden_files(annotation_dir)

# 构建 "VOCdevkit/VOC2007/JPEGImages/" 目录路径
image_dir = os.path.join(work_sapce_dir, "JPEGImages/")
# 如果 "JPEGImages/" 目录不存在，则创建该目录
if not os.path.isdir(image_dir):
    os.mkdir(image_dir)
# 清除 "JPEGImages/" 目录中的隐藏文件
clear_hidden_files(image_dir)

# 构建 "VOCdevkit/VOC2007/YOLOLabels/"目录路径
yolo_labels_dir = os.path.join(work_sapce_dir, "YOLOLabels/")
# 如果 "YOLOLabels/"目录不存在，则创建该目录
if not os.path.isdir(yolo_labels_dir):
    os.mkdir(yolo_labels_dir)
# 清除 "YOLOLabels/" 目录中的隐藏文件
clear_hidden_files(yolo_labels_dir)

# 构建 "VOCdevkit/images/" 目录路径
yolov_images_dir = os.path.join(data_base_dir, "images/")
# 如果 "images/" 目录不存在，则创建该目录
if not os.path.isdir(yolov_images_dir):
    os.mkdir(yolov_images_dir)
# 清除 "images/" 目录中的隐藏文件
clear_hidden_files(yolov_images_dir)

# 构建 "VOCdevkit/labels/" 目录路径
yolov_labels_dir = os.path.join(data_base_dir, "labels/")
# 如果 "labels/" 目录不存在，则创建该目录
if not os.path.isdir(yolov_labels_dir):
    os.mkdir(yolov_labels_dir)
# 清除 "labels/" 目录中的隐藏文件
clear_hidden_files(yolov_labels_dir)

# 构建 "VOCdevkit/images/train/" 目录路径
yolov_images_train_dir = os.path.join(yolov_images_dir, "train/")
# 如果 "train/" 目录不存在，则创建该目录
if not os.path.isdir(yolov_images_train_dir):
    os.mkdir(yolov_images_train_dir)
# 清除 "train/" 目录中的隐藏文件
clear_hidden_files(yolov_images_train_dir)

# 构建 "VOCdevkit/images/val/" 目录路径
yolov_images_test_dir = os.path.join(yolov_images_dir, "val/")
# 如果 "val/" 目录不存在，则创建该目录
if not os.path.isdir(yolov_images_test_dir):
    os.mkdir(yolov_images_test_dir)
# 清除 "val/" 目录中的隐藏文件
clear_hidden_files(yolov_images_test_dir)

# 构建 "VOCdevkit/labels/train/" 目录路径
yolov_labels_train_dir = os.path.join(yolov_labels_dir, "train/")
# 如果 "train/" 目录不存在，则创建该目录
if not os.path.isdir(yolov_labels_train_dir):
    os.mkdir(yolov_labels_train_dir)
# 清除 "train/" 目录中的隐藏文件
clear_hidden_files(yolov_labels_train_dir)

# 构建 "VOCdevkit/labels/val/" 目录路径
yolov_labels_test_dir = os.path.join(yolov_labels_dir, "val/")
# 如果 "val/" 目录不存在，则创建该目录
if not os.path.isdir(yolov_labels_test_dir):
    os.mkdir(yolov_labels_test_dir)
# 清除 "val/" 目录中的隐藏文件
clear_hidden_files(yolov_labels_test_dir)


# 打开（如果文件不存在则创建）一个名为 "yolo_train.txt" 的文件，用于写操作
train_file = open(os.path.join(wd, "yolo_train.txt"), 'w')
# 打开（如果文件不存在则创建）一个名为 "yolo_val.txt" 的文件，用于写操作
test_file = open(os.path.join(wd, "yolo_val.txt"), 'w')

# 关闭 "yolo_train.txt" 文件
train_file.close()
# 关闭 "yolo_val.txt" 文件
test_file.close()

# 以追加模式打开 "yolo_train.txt" 文件
train_file = open(os.path.join(wd, "yolo_train.txt"), 'a')
# 以追加模式打开 "yolo_val.txt" 文件
test_file = open(os.path.join(wd, "yolo_val.txt"), 'a')


# 列出 JPEGImages 目录中的文件
list_imgs = os.listdir(image_dir)

# 生成一个 1 到 100 之间的随机整数
prob = random.randint(1, 100)
print("Probability: %d" % prob)

# 遍历JPEGImages目录中的所有文件
for i in range(0, len(list_imgs)):
    # 构建文件的完整路径
    path = os.path.join(image_dir, list_imgs[i])
    # 检查该路径是否是文件
    if os.path.isfile(path):
        # 获取图像文件的路径
        image_path = image_dir + list_imgs[i]
        # 获取图像文件的名称（不含路径）
        voc_path = list_imgs[i]
        # 获取文件名和扩展名（去除路径）
        (nameWithoutExtention, extention) = os.path.splitext(os.path.basename(image_path))
        (voc_nameWithoutExtention, voc_extention) = os.path.splitext(os.path.basename(voc_path))

        # 构建对应的 annotation 文件名和路径
        annotation_name = nameWithoutExtention + '.xml'
        annotation_path = os.path.join(annotation_dir, annotation_name)

        # 构建对应的 label 文件名和路径
        label_name = nameWithoutExtention + '.txt'
        label_path = os.path.join(yolo_labels_dir, label_name)

    # 再次生成一个 1 到 100 之间的随机整数
    prob = random.randint(1, 100)
    print("Probability: %d" % prob)

    if (prob < TRAIN_RATIO):
        if os.path.exists(annotation_path): # 如果标注文件存在
            train_file.write(image_path + '\n')  # 将图像路径写入训练文件
            convert_annotation(nameWithoutExtention)  # 转换标签
            copyfile(image_path, yolov_images_train_dir + voc_path)  # 复制图像文件到训练图像目录
            copyfile(label_path, yolov_labels_train_dir + label_name)  # 复制标签文件到训练标签目录
    else:  # 测试数据集
        if os.path.exists(annotation_path):  # 如果标注文件存在
            test_file.write(image_path + '\n')  # 将图像路径写入测试文件
            convert_annotation(nameWithoutExtention)  # 转换标签
            copyfile(image_path, yolov_images_test_dir + voc_path)  # 复制图像文件到测试图像目录
            copyfile(label_path, yolov_labels_test_dir + label_name)  # 复制标签文件到测试标签目录
train_file.close()
test_file.close()