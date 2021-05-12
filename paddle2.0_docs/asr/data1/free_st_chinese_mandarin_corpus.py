import argparse
import functools
import os
from utility import download, unpack
from utility import add_arguments, print_arguments

URL_ROOT = 'https://openslr.magicdatatech.com/resources/38'
DATA_URL = URL_ROOT + '/ST-CMDS-20170001_1-OS.tar.gz'
MD5_DATA = 'c28ddfc8e4ebe48949bc79a0c23c5545'

parser = argparse.ArgumentParser(description=__doc__)
add_arg = functools.partial(add_arguments, argparser=parser)
parser.add_argument(
    "--target_dir",
    default="dataset/audio/",
    type=str,
    help="存放音频文件的目录 (默认: %(default)s)")
parser.add_argument(
    "--annotation_text",
    default="dataset/annotation/",
    type=str,
    help="存放音频标注文件的目录 (默认: %(default)s)")
args = parser.parse_args()


def create_annotation_text(data_dir, annotation_path):
    print('Create Free ST-Chinese-Mandarin-Corpus annotation text ...')
    f_a = open(
        os.path.join(annotation_path, 'free_st_chinese_mandarin_corpus.txt'),
        'w',
        encoding='utf-8')
    for subfolder, _, filelist in sorted(os.walk(data_dir)):
        for file in filelist:
            if '.wav' in file:
                file = os.path.join(subfolder, file)
                with open(file[:-4] + '.txt', 'r', encoding='utf-8') as f:
                    line = f.readline()
                f_a.write(file + '\t' + line + '\n')
    f_a.close()


def prepare_dataset(url, md5sum, target_dir, annotation_path):
    """Download, unpack and create manifest file."""
    data_dir = os.path.join(target_dir, 'ST-CMDS-20170001_1-OS')
    if not os.path.exists(data_dir):
        filepath = download(url, md5sum, target_dir)
        unpack(filepath, target_dir)
        os.remove(filepath)
    else:
        print(
            "Skip downloading and unpacking. Free ST-Chinese-Mandarin-Corpus data already exists in %s."
            % target_dir)
    create_annotation_text(data_dir, annotation_path)


def main():
    print_arguments(args)
    if args.target_dir.startswith('~'):
        args.target_dir = os.path.expanduser(args.target_dir)

    prepare_dataset(
        url=DATA_URL,
        md5sum=MD5_DATA,
        target_dir=args.target_dir,
        annotation_path=args.annotation_text)


if __name__ == '__main__':
    main()
