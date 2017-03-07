# Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserved
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os, sys
import cPickle
import numpy as np
from PIL import Image
from optparse import OptionParser

import paddle.utils.image_util as image_util
from py_paddle import swig_paddle, DataProviderConverter
from paddle.trainer.PyDataProvider2 import dense_vector
from paddle.trainer.config_parser import parse_config

import logging
logging.basicConfig(
    format='[%(levelname)s %(asctime)s %(filename)s:%(lineno)s] %(message)s')
logging.getLogger().setLevel(logging.INFO)


def vis_square(data, fname):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    """Take an array of shape (n, height, width) or (n, height, width, 3)
       and visualize each (height, width) thing in a grid of size approx. sqrt(n) by sqrt(n)"""
    # normalize data for display
    data = (data - data.min()) / (data.max() - data.min())
    # force the number of filters to be square
    n = int(np.ceil(np.sqrt(data.shape[0])))
    padding = (
        ((0, n**2 - data.shape[0]), (0, 1),
         (0, 1))  # add some space between filters
        + ((0, 0), ) *
        (data.ndim - 3))  # don't pad the last dimension (if there is one)
    data = np.pad(data, padding, mode='constant',
                  constant_values=1)  # pad with ones (white)
    # tile the filters into an image
    data = data.reshape((n, n) + data.shape[1:]).transpose((0, 2, 1, 3) + tuple(
        range(4, data.ndim + 1)))
    data = data.reshape((n * data.shape[1], n * data.shape[3]) + data.shape[4:])
    plt.imshow(data, cmap='gray')
    plt.savefig(fname)
    plt.axis('off')


class ImageClassifier():
    def __init__(self,
                 train_conf,
                 resize_dim,
                 crop_dim,
                 model_dir=None,
                 use_gpu=True,
                 mean_file=None,
                 oversample=False,
                 is_color=True):
        self.train_conf = train_conf
        self.model_dir = model_dir
        if model_dir is None:
            self.model_dir = os.path.dirname(train_conf)

        self.resize_dim = resize_dim
        self.crop_dims = [crop_dim, crop_dim]
        self.oversample = oversample
        self.is_color = is_color

        self.transformer = image_util.ImageTransformer(is_color=is_color)
        self.transformer.set_transpose((2, 0, 1))
        self.transformer.set_channel_swap((2, 1, 0))

        self.mean_file = mean_file
        if self.mean_file is not None:
            mean = np.load(self.mean_file)['mean']
            mean = mean.reshape(3, self.crop_dims[0], self.crop_dims[1])
            self.transformer.set_mean(mean)  # mean pixel
        else:
            # if you use three mean value, set like:
            # this three mean value is calculated from ImageNet.
            self.transformer.set_mean(np.array([103.939, 116.779, 123.68]))

        conf_args = "use_gpu=%d,is_predict=1" % (int(use_gpu))
        conf = parse_config(train_conf, conf_args)
        swig_paddle.initPaddle("--use_gpu=%d" % (int(use_gpu)))
        self.network = swig_paddle.GradientMachine.createFromConfigProto(
            conf.model_config)
        assert isinstance(self.network, swig_paddle.GradientMachine)
        self.network.loadParameters(self.model_dir)

        dim = 3 * self.crop_dims[0] * self.crop_dims[1]
        slots = [dense_vector(dim)]
        self.converter = DataProviderConverter(slots)

    def get_data(self, img_path):
        """
        1. load image from img_path.
        2. resize or oversampling.
        3. transformer data: transpose, channel swap, sub mean.
        return K x H x W ndarray.

        img_path: image path.
        """
        image = image_util.load_image(img_path, self.is_color)
        # Another way to extract oversampled features is that
        # cropping and averaging from large feature map which is
        # calculated by large size of image.
        # This way reduces the computation.
        if self.oversample:
            image = image_util.resize_image(image, self.resize_dim)
            image = np.array(image)
            input = np.zeros(
                (1, image.shape[0], image.shape[1], 3), dtype=np.float32)
            input[0] = image.astype(np.float32)
            input = image_util.oversample(input, self.crop_dims)
        else:
            image = image.resize(self.crop_dims, Image.ANTIALIAS)
            input = np.zeros(
                (1, self.crop_dims[0], self.crop_dims[1], 3), dtype=np.float32)
            input[0] = np.array(image).astype(np.float32)

        data_in = []
        for img in input:
            img = self.transformer.transformer(img).flatten()
            data_in.append([img.tolist()])
        return data_in

    def forward(self, input_data):
        in_arg = self.converter(input_data)
        return self.network.forwardTest(in_arg)

    def forward(self, data, output_layer):
        input = self.converter(data)
        self.network.forwardTest(input)
        output = self.network.getLayerOutputs(output_layer)
        res = {}
        if isinstance(output_layer, basestring):
            output_layer = [output_layer]
        for name in output_layer:
            # For oversampling, average predictions across crops.
            # If not, the shape of output[name]: (1, class_number),
            # the mean is also applicable.
            res[name] = output[name].mean(0)
        return res


def option_parser():
    usage = "%prog -c config -i data_list -w model_dir [options]"
    parser = OptionParser(usage="usage: %s" % usage)
    parser.add_option(
        "--job",
        action="store",
        dest="job_type",
        choices=[
            'predict',
            'extract',
        ],
        default='predict',
        help="The job type. \
                            predict: predicting,\
                            extract: extract features")
    parser.add_option(
        "--conf",
        action="store",
        dest="train_conf",
        default='models/vgg.py',
        help="network config")
    parser.add_option(
        "--data",
        action="store",
        dest="data_file",
        default='image/dog.png',
        help="image list")
    parser.add_option(
        "--model",
        action="store",
        dest="model_path",
        default=None,
        help="model path")
    parser.add_option(
        "-c", dest="cpu_gpu", action="store_false", help="Use cpu mode.")
    parser.add_option(
        "-g",
        dest="cpu_gpu",
        default=True,
        action="store_true",
        help="Use gpu mode.")
    parser.add_option(
        "--mean",
        action="store",
        dest="mean",
        default='data/mean.meta',
        help="The mean file.")
    parser.add_option(
        "--multi_crop",
        action="store_true",
        dest="multi_crop",
        default=False,
        help="Wether to use multiple crops on image.")
    return parser.parse_args()


def main():
    options, args = option_parser()
    mean = 'data/mean.meta' if not options.mean else options.mean
    conf = 'models/vgg.py' if not options.train_conf else options.train_conf
    obj = ImageClassifier(
        conf,
        32,
        32,
        options.model_path,
        use_gpu=options.cpu_gpu,
        mean_file=mean,
        oversample=options.multi_crop)
    image_path = options.data_file
    if options.job_type == 'predict':
        output_layer = '__fc_layer_2__'
        data = obj.get_data(image_path)
        prob = obj.forward(data, output_layer)
        lab = np.argsort(-prob[output_layer])
        logging.info("Label of %s is: %d", image_path, lab[0])

    elif options.job_type == "extract":
        output_layer = '__conv_0__'
        data = obj.get_data(options.data_file)
        features = obj.forward(data, output_layer)
        dshape = (64, 32, 32)
        fea = features[output_layer].reshape(dshape)
        vis_square(fea, 'fea_conv0.png')


if __name__ == '__main__':
    main()
