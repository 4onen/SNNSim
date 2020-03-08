import yaml


# def seq2num(nums):
#     num = 0
#     numnums = len(nums)
#     for i, n in enumerate(nums):
#         num |= n << (numnums-i-1)

#     return num


seqSize = 20


def pixel2seq(num):
    if num==1:
        return [0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1]
    else:
        return [0,1,0,0,0,0,0,0,1,0,0,1,0,0,0,0,0,0,1,0]

def frame2dict(frame):
    return {'inputs': len(seqSize), 'data': list(map(pixel2seq, frame))}


def frames2yaml(frames):
    return yaml.dump(list(map(frame2dict, frames)))

frames2yaml([[1,1,1,0,0,
              1,0,1,0,0,
              1,0,1,0,0
              1,1,1,0,0,
              0,0,0,0,0],
             [1,1,1,1,1,
              1,0,0,0,1,
              1,0,0,0,1,
              1,0,0,0,1,
              1,1,1,1,1],
             [0,0,0,0,0,
              0,1,1,1,0,
              0,1,0,1,0,
              0,1,1,1,0,
              0,0,0,0,0],
             [1,1,0,0,0,
              0,1,0,0,0,
              0,1,0,0,0,
              0,1,0,0,0,
              1,1,1,0,0],
             [0,1,1,0,0,
              0,0,1,0,0,
              0,0,1,0,0,
              0,0,1,0,0,
              0,1,1,1,0,],
             [0,0,0,1,1,
              0,0,0,0,1,
              0,0,0,0,1,
              0,0,0,0,1,
              0,0,0,1,1,]])