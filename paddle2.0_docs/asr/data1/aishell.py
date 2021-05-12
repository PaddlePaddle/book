import argparse
import os
import functools
from utility import download, unpack
from utility import add_arguments, print_arguments

URL_ROOT = 'https://openslr.magicdatatech.com/resources/33'
DATA_URL = URL_ROOT + '/data_aishell.tgz'
MD5_DATA = '2f494334227864a8a8fec932999db9d8'

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
    print('Create Aishell annotation text ...')
    if not os.path.exists(annotation_path):
        os.makedirs(annotation_path)
    f_a = open(
        os.path.join(annotation_path, 'aishell.txt'), 'w', encoding='utf-8')
    transcript_path = os.path.join(data_dir, 'transcript',
                                   'aishell_transcript_v0.8.txt')
    transcript_dict = {}
    for line in open(transcript_path, 'r', encoding='utf-8'):
        line = line.strip()
        if line == '': continue
        audio_id, text = line.split(' ', 1)
        # remove space
        text = ''.join(text.split())
        transcript_dict[audio_id] = text
    data_types = ['train', 'dev', 'test']
    for type in data_types:
        audio_dir = os.path.join(data_dir, 'wav', type)
        for subfolder, _, filelist in sorted(os.walk(audio_dir)):
            for fname in filelist:
                audio_path = os.path.join(subfolder, fname)
                audio_id = fname[:-4]
                # if no transcription for audio then skipped
                if audio_id not in transcript_dict:
                    continue
                text = transcript_dict[audio_id]
                f_a.write(audio_path + '\t' + text + '\n')
    f_a.close()


def prepare_dataset(url, md5sum, target_dir, annotation_path):
    """Download, unpack and create manifest file."""
    data_dir = os.path.join(target_dir, 'data_aishell')
    if not os.path.exists(data_dir):
        filepath = download(url, md5sum, target_dir)
        unpack(filepath, target_dir)
        # unpack all audio tar files
        audio_dir = os.path.join(data_dir, 'wav')
        for subfolder, _, filelist in sorted(os.walk(audio_dir)):
            for ftar in filelist:
                unpack(os.path.join(subfolder, ftar), subfolder, True)
        os.remove(filepath)
    else:
        print(
            "Skip downloading and unpacking. Aishell data already exists in %s."
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
