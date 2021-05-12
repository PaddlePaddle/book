import Levenshtein as Lev
import paddle


class GreedyDecoder(object):
    def __init__(self, vocabulary, blank_index=0):
        self.int_to_char = dict([(i, c) for (i, c) in enumerate(vocabulary)])
        self.blank_index = blank_index

    # 给定一个数字序列列表，返回相应的字符串
    def convert_to_strings(self,
                           sequences,
                           sizes=None,
                           remove_repetitions=False,
                           return_offsets=False):
        strings = []
        offsets = [] if return_offsets else None
        for x in range(len(sequences)):
            seq_len = sizes[x] if sizes is not None else len(sequences[x])
            string, string_offsets = self.process_string(sequences[x], seq_len,
                                                         remove_repetitions)
            strings.append([string])
            if return_offsets:
                offsets.append([string_offsets])
        if return_offsets:
            return strings, offsets
        else:
            return strings

    # 获取字符，并删除重复的字符
    def process_string(self, sequence, size, remove_repetitions=False):
        string = ""
        offsets = []
        sequence = sequence.numpy()
        for i in range(size):
            char = self.int_to_char[sequence[i].item()]
            if char != self.int_to_char[self.blank_index]:
                # 是否删除重复的字符
                if remove_repetitions and i != 0 and char == self.int_to_char[
                        sequence[i - 1].item()]:
                    pass
                else:
                    string = string + char
                    offsets.append(i)
        return string, paddle.to_tensor(offsets, dtype='int64')

    def cer(self, s1, s2):
        """
       通过计算两个字符串的距离，得出字错率
        """
        s1, s2, = s1.replace(" ", ""), s2.replace(" ", "")
        return Lev.distance(s1, s2)

    def decode(self, probs, sizes=None):
        """
        解码，传入结果的概率解码得到字符串，删除序列中的重复元素和空格。
        """
        max_probs = paddle.argmax(probs, 2)
        strings, offsets = self.convert_to_strings(
            max_probs, sizes, remove_repetitions=True, return_offsets=True)
        return strings, offsets
