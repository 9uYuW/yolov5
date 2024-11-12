# Ultralytics YOLOv5 🚀, AGPL-3.0 license
"""
YOLO-specific modules.

Usage:
    $ python models/yolo.py --cfg yolov5s.yaml
"""
import numpy as np

'''======================1.导入安装好的python库====================='''
import argparse
import contextlib
import math
import os
import platform
import sys
from copy import deepcopy
from pathlib import Path

import torch
import torch.nn as nn

'''===================2.获取当前文件的绝对路径========================'''
FILE = Path(__file__).resolve() # __file__指的是当前文件(即yolo.py),FILE最终保存着当前文件的绝对路径,比如D://yolov5/modles/yolo.py
ROOT = FILE.parents[1]  # YOLOv5 root directory 保存着当前项目的父目录,比如 D://yolov5
if str(ROOT) not in sys.path: # sys.path即当前python环境可以运行的路径,假如当前项目不在该路径中,就无法运行其中的模块,所以就需要加载路径
    sys.path.append(str(ROOT))  # add ROOT to PATH
if platform.system() != "Windows":
    ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative  ROOT设置为相对路径

'''===================3..加载自定义模块============================'''
from models.common import (
    C3,
    C3SPP,
    C3TR,
    SPP,
    SPPF,
    hswish,
    SeModule,
    InvertedResidualBlock,
    DepthSeparableConv,
    Add,
    Bottleneck,
    BottleneckCSP,
    C3Ghost,
    C3x,
    Classify,
    Concat,
    Contract,
    Conv,
    CrossConv,
    DetectMultiBackend,
    DWConv,
    DWConvTranspose2d,
    Expand,
    Focus,
    GhostBottleneck,
    GhostConv,
    Proto, h_sigmoid, h_swish, SELayer, conv_bn_hswish, MobileNetV3,
)
from models.experimental import MixConv2d
from utils.autoanchor import check_anchor_order
from utils.general import LOGGER, check_version, check_yaml, colorstr, make_divisible, print_args
from utils.plots import feature_visualization
from utils.torch_utils import (
    fuse_conv_and_bn,
    initialize_weights,
    model_info,
    profile,
    scale_img,
    select_device,
    time_sync,
)

try:
    import thop  # for FLOPs computation
except ImportError:
    thop = None


class Detect(nn.Module):
    """YOLOv5 Detect head for processing input tensors and generating detection outputs in object detection models."""

    stride = None  # strides computed during build
    dynamic = False  # force grid reconstruction
    export = False  # export mode

    def __init__(self, nc=80, anchors=(), ch=(), inplace=True):
        """Initializes YOLOv5 detection layer with specified classes, anchors, channels, and inplace operations."""
        super().__init__()
        self.nc = nc  # number of classes 数据集类别数量
        # number of outputs per anchor 每个anchor的输出数，前nc个01字符对应类别，后5个对应：是否有目标，目标框的中心，目标框的宽高
        self.no = nc + 5
        self.nl = len(anchors)  # number of detection layers 预测层数，yolov5是3层预测
        # number of anchors anchors的数量，除以2是因为[10,13, 16,30, 33,23]这个长度是6，对应3个anchor
        self.na = len(anchors[0]) // 2
        # init grid 初始化grid列表大小，下面会计算grid，grid就是每个格子的x，y坐标（整数，比如0-19），左上角为(1,1),右下角为(input.w/stride,input.h/stride)
        self.grid = [torch.empty(0) for _ in range(self.nl)]
        self.anchor_grid = [torch.empty(0) for _ in range(self.nl)]  # init anchor grid 初始化anchor_grid列表大小，空列表
        # 注册常量anchor，并将预选框（尺寸）以数对形式存入，并命名为anchors
        self.register_buffer("anchors", torch.tensor(anchors).float().view(self.nl, -1, 2))  # shape(nl,na,2)
        # ch = [80, 40, 20]
        self.m = nn.ModuleList(nn.Conv2d(x, self.no * self.na, 1) for x in ch)  # output conv
        self.inplace = inplace  # use inplace ops (e.g. slice assignment)

    def forward(self, x):
        """Processes input through YOLOv5 layers, altering shape for detection: `x(bs, 3, ny, nx, 85)`."""
        z = []  # inference output
        for i in range(self.nl):
            x[i] = self.m[i](x[i])  # conv
            bs, _, ny, nx = x[i].shape  # x(bs,255,20,20) to x(bs,3,20,20,85)
            x[i] = x[i].view(bs, self.na, self.no, ny, nx).permute(0, 1, 3, 4, 2).contiguous()

            if not self.training:  # inference
                if self.dynamic or self.grid[i].shape[2:4] != x[i].shape[2:4]:
                    self.grid[i], self.anchor_grid[i] = self._make_grid(nx, ny, i)

                if isinstance(self, Segment):  # (boxes + masks)
                    xy, wh, conf, mask = x[i].split((2, 2, self.nc + 1, self.no - self.nc - 5), 4)
                    xy = (xy.sigmoid() * 2 + self.grid[i]) * self.stride[i]  # xy
                    wh = (wh.sigmoid() * 2) ** 2 * self.anchor_grid[i]  # wh
                    y = torch.cat((xy, wh, conf.sigmoid(), mask), 4)
                else:  # Detect (boxes only)
                    # stride: 是一个grid cell的实际尺寸
                    # 经过sigmoid, 值范围变成了(0-1)
                    xy, wh, conf = x[i].sigmoid().split((2, 2, self.nc + 1), 4)
                    # 将值变成范围（-0.5，1.5），为了允许预测框的中心点可以位于当前网格单元格的中心，以及相邻的网格单元格的中心
                    # 注意！旧版代码这里是 xy = (y[..., 0:2] * 2 - 0.5 + self.grid[i]) * self.stride[i]
                    # 新版代码由于grid被调整成以网格中心为原点的坐标系，而不是原来以网格的左上角为原点的坐标系，因此，这里本质上没有变
                    # 具体见笔记
                    xy = (xy * 2 + self.grid[i]) * self.stride[i]  # xy
                    # 标签与anchor的宽高之间的比例在训练时被设置为[0.25, 4]的范围
                    wh = (wh * 2) ** 2 * self.anchor_grid[i]  # wh
                    y = torch.cat((xy, wh, conf), 4)
                # 存储每个特征图检测框的信息
                z.append(y.view(bs, self.na * nx * ny, self.no))
        # 训练阶段直接返回x
        # 预测阶段返回3个特征图拼接的结果
        return x if self.training else (torch.cat(z, 1),) if self.export else (torch.cat(z, 1), x)
    """
    这段代码主要是对三个feature map分别进行处理：(n, 255, 80, 80),(n, 255, 40, 40),(n, 255, 20, 20)
    首先进行for循环，每次i的循环，产生一个z。维度重排列：(n, 255, _, _) -> (n, 3, nc+5, ny, nx) -> (n, 3, ny, nx, nc+5)，三层分别预测了80*80、40*40、20*20次。
    接着构造网格，因为推理返回的不是归一化后的网格偏移量，需要再加上网格的位置，得到最终的推理坐标，再送入nms。所以这里构建网格就是为了记录每个grid的网格坐标方便后面使用
    最后按损失函数的回归方式来转换坐标，利用sigmoid激活函数计算定位参数，cat(dim=1)为直接拼接。注意： 训练阶段直接返回x ，而预测阶段返回3个特征图拼接的结果
    """

    def _make_grid(self, nx=20, ny=20, i=0, torch_1_10=check_version(torch.__version__, "1.10.0")):
        """Generates a mesh grid for anchor boxes with optional compatibility for torch versions < 1.10."""
        d = self.anchors[i].device
        t = self.anchors[i].dtype
        shape = 1, self.na, ny, nx, 2  # grid shape
        y, x = torch.arange(ny, device=d, dtype=t), torch.arange(nx, device=d, dtype=t)
        """
        yv: shape: (ny, nx)
        tensor([[1, 1, ...1],
                [2, 2, ...2],
                ...,
                [ny, ny, ...ny]])
        xv: shape: (ny, nx)
        tensor([[1, 2, ...nx],
                [1, 2, ...nx],
                ...,
                [1, 2, ...nx]])
        """
        yv, xv = torch.meshgrid(y, x, indexing="ij") if torch_1_10 else torch.meshgrid(y, x)  # torch>=0.7 compatibility
        """
        after stack:[[[[1,1],[1,2],...[1,nx]],
                    [[2,1],[2,2],...[2,nx]],
                                ...,
                    [[ny,1],[ny,2],...[ny,nx]]]]
        shape: (ny, nx, 2)
        """
        # grid --> (20, 20, 2), 复制成3倍，因为是三个框 -> (1, 3, 20, 20, 2)
        grid = torch.stack((xv, yv), 2).expand(shape) - 0.5  # add grid offset, i.e. y = 2.0 * x - 0.5
        # anchor_grid是每个anchor宽高。
        anchor_grid = (self.anchors[i] * self.stride[i]).view((1, self.na, 1, 1, 2)).expand(shape)
        return grid, anchor_grid


class Segment(Detect):
    """YOLOv5 Segment head for segmentation models, extending Detect with mask and prototype layers."""

    def __init__(self, nc=80, anchors=(), nm=32, npr=256, ch=(), inplace=True):
        """Initializes YOLOv5 Segment head with options for mask count, protos, and channel adjustments."""
        super().__init__(nc, anchors, ch, inplace)
        self.nm = nm  # number of masks
        self.npr = npr  # number of protos
        self.no = 5 + nc + self.nm  # number of outputs per anchor
        self.m = nn.ModuleList(nn.Conv2d(x, self.no * self.na, 1) for x in ch)  # output conv
        self.proto = Proto(ch[0], self.npr, self.nm)  # protos
        self.detect = Detect.forward

    def forward(self, x):
        """Processes input through the network, returning detections and prototypes; adjusts output based on
        training/export mode.
        """
        p = self.proto(x[0])
        x = self.detect(self, x)
        return (x, p) if self.training else (x[0], p) if self.export else (x[0], p, x[1])


class BaseModel(nn.Module):
    """YOLOv5 base model."""

    def forward(self, x, profile=False, visualize=False):
        """Executes a single-scale inference or training pass on the YOLOv5 base model, with options for profiling and
        visualization.
        """
        return self._forward_once(x, profile, visualize)  # single-scale inference, train

    def _forward_once(self, x, profile=False, visualize=False):
        """Performs a forward pass on the YOLOv5 model, enabling profiling and feature visualization options."""
        # 各网络层输出, 各网络层推导耗时
        # y: 存放着self.save=True的每一层的输出，因为后面的层结构concat等操作要用到
        # dt: 在profile中做性能评估时使用
        y, dt = [], []  # outputs
        for m in self.model:
            # if isinstance(x, torch.Tensor):
            #     print(x.shape)
            if m.f != -1:  # if not from previous layer
                x = y[m.f] if isinstance(m.f, int) else [x if j == -1 else y[j] for j in m.f]  # from earlier layers
            # 测试该网络层的性能
            if profile:
                self._profile_one_layer(m, x, dt)
            # 使用该网络层进行推导, 得到该网络层的输出
            x = m(x)  # run
            # 存放着self.save的每一层的输出，因为后面需要用来作concat等操作要用到  不在self.save层的输出就为None
            y.append(x if m.i in self.save else None)  # save output
            # 将每一层的输出结果保存到y
            if visualize:
                feature_visualization(x, m.type, m.i, save_dir=visualize)
        return x

    # ===6._profile_one_layer（）:打印日志信息=== #
    def _profile_one_layer(self, m, x, dt):
        """Profiles a single layer's performance by computing GFLOPs, execution time, and parameters."""
        c = m == self.model[-1]  # is final layer, copy input as inplace fix
        o = thop.profile(m, inputs=(x.copy() if c else x,), verbose=False)[0] / 1e9 * 2 if thop else 0  # FLOPs
        t = time_sync()
        for _ in range(10):
            m(x.copy() if c else x)
        dt.append((time_sync() - t) * 100)
        if m == self.model[0]:
            LOGGER.info(f"{'time (ms)':>10s} {'GFLOPs':>10s} {'params':>10s}  module")
        LOGGER.info(f"{dt[-1]:10.2f} {o:10.2f} {m.np:10.0f}  {m.type}")
        if c:
            LOGGER.info(f"{sum(dt):10.2f} {'-':>10s} {'-':>10s}  Total")

    def fuse(self):
        """Fuses Conv2d() and BatchNorm2d() layers in the model to improve inference speed."""
        LOGGER.info("Fusing layers... ")
        for m in self.model.modules():
            # 如果当前层是卷积层Conv且有bn结构, 那么就调用fuse_conv_and_bn函数讲conv和bn进行融合, 加速推理
            if isinstance(m, (Conv, DWConv)) and hasattr(m, "bn"):
                m.conv = fuse_conv_and_bn(m.conv, m.bn)  # update conv
                delattr(m, "bn")  # remove batchnorm
                m.forward = m.forward_fuse  # update forward
        self.info()
        return self

    # ===11.info():打印模型结构信息=== #
    def info(self, verbose=False, img_size=640):
        """Prints model information given verbosity and image size, e.g., `info(verbose=True, img_size=640)`."""
        model_info(self, verbose, img_size)

    # ===12._apply():将模块转移到 CPU/ GPU上=== #
    def _apply(self, fn):
        """Applies transformations like to(), cpu(), cuda(), half() to model tensors excluding parameters or registered
        buffers.
        """
        self = super()._apply(fn)
        m = self.model[-1]  # Detect()
        if isinstance(m, (Detect, Segment)):
            m.stride = fn(m.stride)
            m.grid = list(map(fn, m.grid))
            if isinstance(m.anchor_grid, list):
                m.anchor_grid = list(map(fn, m.anchor_grid))
        return self


class DetectionModel(BaseModel):
    """YOLOv5 detection model class for object detection tasks, supporting custom configurations and anchors."""

    def __init__(self, cfg="yolov5s.yaml", ch=3, nc=None, anchors=None):
        """Initializes YOLOv5 model with configuration file, input channels, number of classes, and custom anchors."""
        super().__init__()
        # 检查传入的参数格式，如果cfg是字典格式
        if isinstance(cfg, dict):
            self.yaml = cfg  # model dict
        # 若不是字典 则为yaml文件路径
        else:  # is *.yaml
            import yaml  # for torch hub
            # 保存文件名
            self.yaml_file = Path(cfg).name
            with open(cfg, encoding="ascii", errors="ignore") as f:
                # 将yaml文件加载为字典
                self.yaml = yaml.safe_load(f)  # model dict

        # Define model
        # yaml.get('ch', ch)表示若不存在键'ch',则返回值ch
        ch = self.yaml["ch"] = self.yaml.get("ch", ch)  # input channels

        # 判断类的通道数和yaml中的通道数是否相等
        if nc and nc != self.yaml["nc"]:
            # 在终端给出提示
            LOGGER.info(f"Overriding model.yaml nc={self.yaml['nc']} with nc={nc}")
            # 将yaml中的值修改为构造方法中的值
            self.yaml["nc"] = nc  # override yaml value
        if anchors:
            LOGGER.info(f"Overriding model.yaml anchors with anchors={anchors}")
            self.yaml["anchors"] = round(anchors)  # override yaml value

        # 解析模型，self.model是解析后的模型 self.save是每一层与之相连的层
        self.model, self.save = parse_model(deepcopy(self.yaml), ch=[ch])  # model, savelist
        # 加载每一类的类别名（编号）
        self.names = [str(i) for i in range(self.yaml["nc"])]  # default names
        # inplace指的是原地操作 如x+=1 有利于节约内存，默认True
        self.inplace = self.yaml.get("inplace", True)

        # Build strides, anchors
        m = self.model[-1]  # Detect()
        if isinstance(m, (Detect, Segment)):

            def _forward(x):
                """Passes the input 'x' through the model and returns the processed output."""
                return self.forward(x)[0] if isinstance(m, Segment) else self.forward(x)
            # 定义一个256 * 256大小的输入（无实际意义，仅是为了模拟输入，测量stride）
            s = 320  # 2x min stride
            m.inplace = self.inplace
            # 将[1, ch, 256, 256]大小的tensor进行一次向前传播，得到3层的输出，用输入大小256分别除以输出大小得到每一层的下采样倍数stride
            m.stride = torch.tensor([s / x.shape[-2] for x in _forward(torch.zeros(1, ch, s, s))])  # forward
            # 检查anchor顺序与stride顺序是否一致
            check_anchor_order(m)
            # 分别用最初的anchor大小除以stride将anchor线性缩放到对应层上
            m.anchors /= m.stride.view(-1, 1, 1)
            # 将步长保存至模型
            self.stride = m.stride
            # 初始化bias
            self._initialize_biases()  # only run once

        # Init weights, biases
        # 初始化权重
        initialize_weights(self)
        # 打印模型信息
        self.info()
        LOGGER.info("")

    def forward(self, x, augment=False, profile=False, visualize=False):
        """Performs single-scale or augmented inference and may include profiling or visualization."""
        if augment:
            # 增强训练，对数据采取了一些了操作
            return self._forward_augment(x)  # augmented inference, None
        # 默认执行，正常前向推理
        return self._forward_once(x, profile, visualize)  # single-scale inference, train

    """
    设分类数为 80 、检测框属性数为 5，则基本步骤是：
    对图像进行变换：总共 3 次，分别是 [ 原图 ]，[ 尺寸缩小到原来的 0.83，同时水平翻转 ]，[ 尺寸缩小到原来的 0.67 ]
    对图像使用 _forward_once 函数，得到在 eval 模式下网络模型的推导结果。对原图是 shape 为 [1, 22743, 85] 的图像检测框信息 (见 Detect 对象的 forward 函数)
    根据 尺寸缩小倍数、翻转维度 对检测框信息进行逆变换，添加进列表 y
    截取 y[0] 对大物体的检测结果，保留 y[1] 所有的检测结果，截取 y[2] 对小物体的检测结果，拼接得到新的检测框信息
    """
    def _forward_augment(self, x):
        """Performs augmented inference across different scales and flips, returning combined detections."""
        # 这个函数只在 val、detect 主函数中使用，用于提高推导的精度。
        img_size = x.shape[-2:]  # height, width
        s = [1, 0.83, 0.67]  # scales
        f = [None, 3, None]  # flips (2-ud, 3-lr)
        y = []  # outputs
        for si, fi in zip(s, f):
            # scale_img函数的作用就是根据传入的参数缩放和翻转图像
            xi = scale_img(x.flip(fi) if fi else x, si, gs=int(self.stride.max()))
            yi = self._forward_once(xi)[0]  # forward
            # cv2.imwrite(f'img_{si}.jpg', 255 * xi[0].cpu().numpy().transpose((1, 2, 0))[:, :, ::-1])  # save
            #  恢复数据增强前的模样
            yi = self._descale_pred(yi, fi, si, img_size)
            y.append(yi)
        # 对不同尺寸进行不同程度的筛选
        y = self._clip_augmented(y)  # clip augmented tails
        return torch.cat(y, 1), None  # augmented inference, train

    # 将推理结果恢复到原图尺寸(逆操作)
    def _descale_pred(self, p, flips, scale, img_size):
        """De-scales predictions from augmented inference, adjusting for flips and image size."""
        if self.inplace:
            p[..., :4] /= scale  # de-scale
            if flips == 2:
                p[..., 1] = img_size[0] - p[..., 1]  # de-flip ud
            elif flips == 3:
                p[..., 0] = img_size[1] - p[..., 0]  # de-flip lr
        else:
            x, y, wh = p[..., 0:1] / scale, p[..., 1:2] / scale, p[..., 2:4] / scale  # de-scale
            if flips == 2:
                y = img_size[0] - y  # de-flip ud
            elif flips == 3:
                x = img_size[1] - x  # de-flip lr
            p = torch.cat((x, y, wh, p[..., 4:]), -1)
        return p

    def _clip_augmented(self, y):
        """Clips augmented inference tails for YOLOv5 models, affecting first and last tensors based on grid points and
        layer counts.
        """
        nl = self.model[-1].nl  # number of detection layers (P3-P5)
        g = sum(4**x for x in range(nl))  # grid points
        e = 1  # exclude layer count
        i = (y[0].shape[1] // g) * sum(4**x for x in range(e))  # indices
        y[0] = y[0][:, :-i]  # large
        i = (y[-1].shape[1] // g) * sum(4 ** (nl - 1 - x) for x in range(e))  # indices
        y[-1] = y[-1][:, i:]  # small
        return y

    def _initialize_biases(self, cf=None):
        """
        Initializes biases for YOLOv5's Detect() module, optionally using class frequencies (cf).

        For details see https://arxiv.org/abs/1708.02002 section 3.3.
        """
        # cf = torch.bincount(torch.tensor(np.concatenate(dataset.labels, 0)[:, 0]).long(), minlength=nc) + 1.
        m = self.model[-1]  # Detect() module
        for mi, s in zip(m.m, m.stride):  # from
            b = mi.bias.view(m.na, -1)  # conv.bias(255) to (3,85)
            b.data[:, 4] += math.log(8 / (640 / s) ** 2)  # obj (8 objects per 640 image)
            b.data[:, 5 : 5 + m.nc] += (
                math.log(0.6 / (m.nc - 0.99999)) if cf is None else torch.log(cf / cf.sum())
            )  # cls
            mi.bias = torch.nn.Parameter(b.view(-1), requires_grad=True)


Model = DetectionModel  # retain YOLOv5 'Model' class for backwards compatibility


class SegmentationModel(DetectionModel):
    """YOLOv5 segmentation model for object detection and segmentation tasks with configurable parameters."""

    def __init__(self, cfg="yolov5s-seg.yaml", ch=3, nc=None, anchors=None):
        """Initializes a YOLOv5 segmentation model with configurable params: cfg (str) for configuration, ch (int) for channels, nc (int) for num classes, anchors (list)."""
        super().__init__(cfg, ch, nc, anchors)


class ClassificationModel(BaseModel):
    """YOLOv5 classification model for image classification tasks, initialized with a config file or detection model."""

    def __init__(self, cfg=None, model=None, nc=1000, cutoff=10):
        """Initializes YOLOv5 model with config file `cfg`, input channels `ch`, number of classes `nc`, and `cuttoff`
        index.
        """
        super().__init__()
        self._from_detection_model(model, nc, cutoff) if model is not None else self._from_yaml(cfg)

    def _from_detection_model(self, model, nc=1000, cutoff=10):
        """Creates a classification model from a YOLOv5 detection model, slicing at `cutoff` and adding a classification
        layer.
        """
        if isinstance(model, DetectMultiBackend):
            model = model.model  # unwrap DetectMultiBackend
        model.model = model.model[:cutoff]  # backbone
        m = model.model[-1]  # last layer
        ch = m.conv.in_channels if hasattr(m, "conv") else m.cv1.conv.in_channels  # ch into module
        c = Classify(ch, nc)  # Classify()
        c.i, c.f, c.type = m.i, m.f, "models.common.Classify"  # index, from, type
        model.model[-1] = c  # replace
        self.model = model.model
        self.stride = model.stride
        self.save = []
        self.nc = nc

    def _from_yaml(self, cfg):
        """Creates a YOLOv5 classification model from a specified *.yaml configuration file."""
        self.model = None

"""
    parse_model函数用在DetectionModel模块中，主要作用是解析模型yaml的模块，
    通过读取yaml文件中的配置，并且到common.py中找到相对于的模块，然后组成一个完整的模型解析模型文件(字典形式)，并搭建网络结构。
    简单来说，就是把yaml文件中的网络结构实例化成对应的模型。后续如果需要动模型框架的话，需要对这个函数做相应的改动。
    Args:
        d:  yaml 配置文件（字典形式），yolov5s.yaml中的6个元素 + ch
        ch:  记录模型每一层的输出channel，初始ch=[3]，后面会删除
        na：  判断anchor的数量
        no：  根据anchor数量推断的输出维度
"""
def parse_model(d, ch):
    '''===================1. 获取对应参数============================'''
    """Parses a YOLOv5 model from a dict `d`, configuring layers based on input channels `ch` and model architecture."""
    # 打印表头：from n params module arguments 等信息
    LOGGER.info(f"\n{'':>3}{'from':>18}{'n':>3}{'params':>10}  {'module':<40}{'arguments':<30}")
    anchors, nc, gd, gw, act, ch_mul = (
        d["anchors"],
        d["nc"],
        d["depth_multiple"],
        d["width_multiple"],
        d.get("activation"),
        d.get("channel_multiple"),
    )
    if act:
        Conv.default_act = eval(act)  # redefine default activation, i.e. Conv.default_act = nn.SiLU()
        LOGGER.info(f"{colorstr('activation:')} {act}")  # print
    if not ch_mul:
        ch_mul = 8
    # na: 每组先验框包含的先验框数
    na = (len(anchors[0]) // 2) if isinstance(anchors, list) else anchors  # number of anchors
    # no: na * 属性数 (5 + 分类数)
    no = na * (nc + 5)  # number of outputs = anchors * (classes + 5)

    '''===================2. 搭建网络前准备============================'''
    """
        layers:   保存每一层的层结构
        save:   记录下所有层结构中from不是-1的层结构序号
        c2:   保存当前层的输出channel 
        f： from，当前层输入来自哪些层
        n： number，当前层次数 初定
        m： module，当前层类别
        args： 当前层类参数 初定
    """
    # 网络单元列表, 网络输出引用列表, 当前的输出通道数
    layers, save, c2 = [], [], ch[-1]  # layers, savelist, ch out
    inputsz = 320

    # 读取 backbone, head 中的网络单元
    for i, (f, n, m, args) in enumerate(d["backbone"] + d["head"]):  # from, number, module, args
        # 利用 eval 函数, 读取 model 参数对应的类名 如‘Focus’,'Conv'等
        m = eval(m) if isinstance(m, str) else m  # eval strings

        # 利用 eval 函数将字符串转换为变量 如‘None’,‘nc’，‘anchors’等
        for j, a in enumerate(args):
            with contextlib.suppress(NameError):
                args[j] = eval(a) if isinstance(a, str) else a  # eval strings

        '''===================3. 更新当前层的参数，计算c2============================'''
        # depth gain: 控制深度，如yolov5s: n*0.33
        n = n_ = max(round(n * gd), 1) if n > 1 else n  # depth gain
        if m in {
            Conv,
            GhostConv,
            Bottleneck,
            GhostBottleneck,
            SPP,
            SPPF,
            DWConv,
            MixConv2d,
            Focus,
            CrossConv,
            BottleneckCSP,
            C3,
            C3TR,
            C3SPP,
            C3Ghost,
            nn.ConvTranspose2d,
            DWConvTranspose2d,
            C3x,
            h_sigmoid,
            h_swish,
            SELayer,
            conv_bn_hswish,
            MobileNetV3
        }:
            # c1: 当前层的输入channel数; c2: 当前层的输出channel数(初定); ch: 记录着所有层的输出channel数
            c1, c2 = ch[f], args[0]
            # 只有最后一层c2=no，最后一层不用控制宽度，输出channel必须是no
            if c2 != no:  # if not output
                # width gain: 控制宽度，如yolov5s: c2*0.5; c2: 当前层的最终输出channel数(间接控制宽度)
                """
                make_divisible（）代码如下：   
               # 使得X能够被divisor整除
                 def make_divisible(x, divisor):
                     return math.ceil(x / divisor) * divisor
                """
                c2 = make_divisible(c2 * gw, ch_mul)

            '''===================4.使用当前层的参数搭建当前层============================'''
            # 在初始args的基础上更新，加入当前层的输入channel并更新当前层
            args = [c1, c2, *args[1:]]
            # 如果当前层是BottleneckCSP/C3/C3TR/C3Ghost/C3x，则需要在args中加入Bottleneck的个数
            # [in_channels, out_channels, Bottleneck个数, Bool(shortcut有无标记)]
            if m in {BottleneckCSP, C3, C3TR, C3Ghost, C3x}:
                args.insert(2, n)  # number of repeats
                # 恢复默认值1
                n = 1
        elif m is InvertedResidualBlock:
            c1 = ch[f]
            c2 = args[1]
            args = [c1, *args[0:]]
        elif m is DepthSeparableConv:
            c2 = args[0]
            c1 = ch[f]
            args = [c1, *args[0:]]
        # 判断是否是归一化模块
        elif m is nn.BatchNorm2d:
            # BN层只需要返回上一层的输出channel
            args = [ch[f]]
        # 判断是否是tensor连接模块
        elif m is Concat:
            # Concat层则将f中所有的输出累加得到这层的输出channel
            c2 = sum(ch[x] for x in f)
        elif m is Add:
            c2 = ch[f[0]]
        # TODO: channel, gw, gd
        # 判断是否是detect模块
        elif m in {Detect, Segment}:
            # 在args中加入三个Detect层的输出channel [nc, anchors, [128, 256, 512]]
            # Segment [nc, anchors, 32, 256, [128, 256, 512]]
            args.append([ch[x] for x in f])
            if isinstance(args[1], int):  # number of anchors
                args[1] = [list(range(args[1] * 2))] * len(f)
            if m is Segment:
                args[3] = make_divisible(args[3] * gw, ch_mul)
        elif m is Contract:
            c2 = ch[f] * args[0] ** 2
        elif m is Expand:
            c2 = ch[f] // args[0] ** 2
        else:
            c2 = ch[f]
        """
            经过以上处理，args里面保存的前两个参数就是module的输入通道数、输出通道数。
            只有BottleneckCSP和C3这两种module会根据深度参数n调整该模块的重复迭加次数。
            然后进行的是其他几种类型的Module判断：
            • 如果是BN层，只需要返回上一层的输出channel，通道数保持不变。
            • 如果是Concat层，则将f中所有的输出累加得到这层的输出channel，f是所有需要拼接层的index，输出通道c2是所有层的和。
            • 如果是Detect层，则对应检测头部分，这块下一小节细讲。
            Contract和Expand目前未在模型中使用。
        """

        '''===================5.打印和保存layers信息============================'''
        # m_: 得到当前层的module，将n个模块组合存放到m_里面
        m_ = nn.Sequential(*(m(*args) for _ in range(n))) if n > 1 else m(*args)  # module

        # 打印当前层结构的一些基本信息
        t = str(m)[8:-2].replace("__main__.", "")  # module type

        # 计算这一层的参数量
        np = sum(x.numel() for x in m_.parameters())  # number params
        m_.i, m_.f, m_.type, m_.np = i, f, t, np  # attach index, 'from' index, type, number params
        LOGGER.info(f"{i:>3}{str(f):>18}{n_:>3}{np:10.0f}  {t:<40}{str(args):<30}")  # print

        # 把所有层结构中的from不是-1的值记下 [6,4,14,10,17,20,23]
        save.extend(x % i for x in ([f] if isinstance(f, int) else f) if x != -1)  # append to savelist
        layers.append(m_)
        if i == 0:
            ch = [] # 去除输入channel[3]
        ch.append(c2)
    return nn.Sequential(*layers), sorted(save)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", type=str, default="yolov5s.yaml", help="model.yaml")
    parser.add_argument("--batch-size", type=int, default=1, help="total batch size for all GPUs")
    parser.add_argument("--device", default="", help="cuda device, i.e. 0 or 0,1,2,3 or cpu")
    parser.add_argument("--profile", action="store_true", help="profile model speed")
    parser.add_argument("--line-profile", action="store_true", help="profile model speed layer by layer")
    parser.add_argument("--test", action="store_true", help="test all yolo*.yaml")
    opt = parser.parse_args()
    opt.cfg = check_yaml(opt.cfg)  # check YAML
    print_args(vars(opt))
    device = select_device(opt.device)

    # Create model
    im = torch.rand(opt.batch_size, 3, 640, 640).to(device)
    model = Model(opt.cfg).to(device)

    # Options
    if opt.line_profile:  # profile layer by layer
        model(im, profile=True)

    elif opt.profile:  # profile forward-backward
        results = profile(input=im, ops=[model], n=3)

    elif opt.test:  # test all models
        for cfg in Path(ROOT / "models").rglob("yolo*.yaml"):
            try:
                _ = Model(cfg)
            except Exception as e:
                print(f"Error in {cfg}: {e}")

    else:  # report fused model summary
        model.fuse()
