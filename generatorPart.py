import yaml
# def seq2num(nums):
#     num = 0
#     numnums = len(nums)
#     for i, n in enumerate(nums):
#         num |= n << (numnums-i-1)

#     return num


seqSize = 20


def pixel2seq(num):
    if num == 1:
        return bin(0b01010101010101010101)
    else:
        return bin(0b01000000100100000010)


def frame2dict(frame):
    return {'inputs': seqSize, 'data': list(map(pixel2seq, frame))}


def frames2yaml(frames):
    file = open('benchmarks/generatedInput.yaml', 'w')
    return yaml.dump(list(map(frame2dict, frames)), file)


frames2yaml([[1, 1, 1, 0, 0,
              1, 0, 1, 0, 0,
              1, 0, 1, 0, 0,
              1, 1, 1, 0, 0,
              0, 0, 0, 0, 0],
             [1, 1, 1, 1, 1,
              1, 0, 0, 0, 1,
              1, 0, 0, 0, 1,
              1, 0, 0, 0, 1,
              1, 1, 1, 1, 1],
             [0, 0, 0, 0, 0,
              0, 1, 1, 1, 0,
              0, 1, 0, 1, 0,
              0, 1, 1, 1, 0,
              0, 0, 0, 0, 0],
             [1, 1, 0, 0, 0,
              0, 1, 0, 0, 0,
              0, 1, 0, 0, 0,
              0, 1, 0, 0, 0,
              1, 1, 1, 0, 0],
             [0, 1, 1, 0, 0,
              0, 0, 1, 0, 0,
              0, 0, 1, 0, 0,
              0, 0, 1, 0, 0,
              0, 1, 1, 1, 0, ],
             [0, 0, 0, 1, 1,
              0, 0, 0, 0, 1,
              0, 0, 0, 0, 1,
              0, 0, 0, 0, 1,
              0, 0, 0, 1, 1, ]])
