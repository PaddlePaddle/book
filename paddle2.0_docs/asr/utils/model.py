import paddle
import paddle.nn as nn
from paddle.nn.initializer import KaimingNormal


# 门控线性单元 Gated Linear Units (GLU)
class GLU(nn.Layer):
    def __init__(self, axis):
        super(GLU, self).__init__()
        self.sigmoid = nn.Sigmoid()
        self.axis = axis

    def forward(self, x):
        a, b = paddle.split(x, num_or_sections=2, axis=self.axis)
        act_b = self.sigmoid(b)
        out = paddle.multiply(x=a, y=act_b)
        return out


# 基本卷积块
class ConvBlock(nn.Layer):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride,
                 padding=0,
                 p=0.5):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv1D(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            weight_attr=KaimingNormal())
        self.conv = nn.utils.weight_norm(self.conv)
        self.act = GLU(axis=1)
        self.dropout = nn.Dropout(p)

    def forward(self, x):
        x = self.conv(x)
        x = self.act(x)
        x = self.dropout(x)
        return x


# PPASR模型
class PPASR(nn.Layer):
    def __init__(self, vocabulary, data_mean=None, data_std=None, name="PPASR"):
        super(PPASR, self).__init__(name_scope=name)
        # 数据均值和标准值到模型中，方便以后推理使用
        if data_mean is None:
            data_mean = paddle.to_tensor(1.0)
        if data_std is None:
            data_std = paddle.to_tensor(1.0)
        self.register_buffer("data_mean", data_mean, persistable=True)
        self.register_buffer("data_std", data_std, persistable=True)
        # 模型的输出大小，字典大小+1
        self.output_units = len(vocabulary) + 1
        self.conv1 = ConvBlock(128, 500, 48, 2, padding=97, p=0.2)
        self.conv2 = ConvBlock(250, 500, 7, 1, p=0.3)
        self.conv3 = ConvBlock(250, 2000, 32, 1, p=0.3)
        self.conv4 = ConvBlock(1000, 2000, 1, 1, p=0.3)
        self.out = nn.utils.weight_norm(
            nn.Conv1D(1000, self.output_units, 1, 1))

    def forward(self, x, input_lens=None):
        x = self.conv1(x)
        for i in range(7):
            x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.out(x)
        if input_lens is not None:
            return x, paddle.to_tensor(input_lens / 2 + 1, dtype='int64')
        return x
