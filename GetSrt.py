'''
Description:
Author: Hejun Jiang
Date: 2020-11-19 14:54:17
LastEditTime: 2020-11-23 15:02:30
LastEditors: Hejun Jiang
Version: v0.0.1
Contact: jianghejun@hccl.ioa.ac.cn
Corporation: hccl
'''
# -*- coding: utf-8 -*-
import os
import time
import cv2  # BGR
import shutil
import numpy as np
from config import *
from PIL import Image  # RGB
from model import text_predict, crnn_handle
from difflib import SequenceMatcher
from apphelper.image import base64_to_PIL


def strdiff(str1, str2):
    return SequenceMatcher(None, str1, str2).ratio()


def conut_chinese(strs):
    n = 0
    for _char in strs:
        if '\u4e00' <= _char <= '\u9fa5':
            n += 1
            if n >= SubMinLen:
                return True
    return False


def GetSubImg(img):
    hight = img.shape[0]
    width = img.shape[1]
    subtitleimg = img[int(
        hight * SubUpRatio): int(hight * SubBottomRatio), int(
        width * SubLeftRatio): int(width * SubRightRatio)]
    return subtitleimg


def GetSrtFromVideo(VideoDir, JumpFrame=0, isSaveImg=False):
    dirname = os.path.dirname(VideoDir)
    basename = os.path.basename(VideoDir)
    video = cv2.VideoCapture(VideoDir)
    fps = video.get(cv2.CAP_PROP_FPS)  # fps
    framenum = video.get(cv2.CAP_PROP_FRAME_COUNT)  # frame number
    totaltime = framenum / fps  # video length, s
    srtpath = os.path.join(
        dirname, basename.split('.')[0] + '.srt')
    saveimgdir = os.path.join(
        dirname, basename.split('.')[0])
    if os.path.exists(srtpath):
        os.remove(srtpath)
    if os.path.isdir(saveimgdir):
        shutil.rmtree(saveimgdir)
    os.makedirs(saveimgdir)
    print('video path:', VideoDir)
    print('srt path:', srtpath)
    print('total frame num:%d,' % framenum, 'fps:%d' %
          fps, 'timeLen:%.2fs' % totaltime)

    idx = 0
    result = []
    while True:
        success, img = video.read()
        milltime = video.get(cv2.CAP_PROP_POS_MSEC)
        if not success:
            break
        idx += 1
        if idx % (JumpFrame + 1) != 0:
            continue
        subimg = GetSubImg(img)
        text = crnn_handle.predict(Image.fromarray(
            subimg).convert("RGB").convert('L'))  # 识别的文本
        result.append([text, milltime])
        print('text:', text)
    print('get results done, result length:', len(result))

    # f = open('./temp.txt', 'w', encoding='utf-8')
    # for item in result:
    #     f.write(item[0] + ' || ' + str(item[1]) + '\n')
    # f.close()

    # f = open('./temp.txt', 'r', encoding='utf-8')
    # lis = f.readlines()
    # f.close()
    # result = []
    # for line in lis:
    #     l = line.split(' || ')
    #     result.append([l[0], float(l[1])])

    i = 0  # result的开始
    idx = 1
    while True:
        if i >= len(result):
            break
        if conut_chinese(result[i][0]):
            start = result[i]
            print('start:', start)
            # 从start idx开始之后的SubMaxTime秒，包括start idx在内的SubMaxTime秒张图
            rlist = list(reversed(result[i: int(SubMaxTime * fps + i)]))

            for j, ritem in enumerate(rlist):
                if conut_chinese(ritem[0]) and strdiff(ritem[0], start[0]) >= SimilarThreshold:
                    end = ritem
                    fakei = i + (len(rlist) - j - 1)
                    if result[fakei][1] - result[i][1] >= DiffTime*1000:
                        fp = open(srtpath, 'a+', encoding='gb2312')  # chinese
                        fp.write(str(idx) + '\n')
                        fp.write(time.strftime("%H:%M:%S", time.localtime(start[1]/1000)) + ',' + str(int(start[1] % 1000)) +
                                 ' --> ' + time.strftime("%H:%M:%S", time.localtime(end[1]/1000)) + ',' + str(int(end[1] % 1000)) + '\n')
                        fp.write(start[0] + '\n\n')
                        fp.close()
                        idx += 1
                        i = fakei  # 满足条件，比较长时间
                        break  # 匹配完成一次就退出
        i += 1


def videosDetect():
    '''for videos detect'''
    vtype = ['mp4', 'mkv']
    t = time.time()
    videosdir = './test_videos/'
    for dirpath, dirname, dirfile in os.walk(videosdir):
        for file in dirfile:
            if file.split('.')[-1] in vtype:
                GetSrtFromVideo(VideoDir=os.path.join(
                    dirpath, file), JumpFrame=1, isSaveImg=True)
    print('total spend time:%d s\n' % (time.time() - t))


def imagesDetect():
    '''for images detect'''
    imgtype = ['png', 'jpg', 'jpeg']
    t = time.time()
    imgdir = './test_imgs/'
    for dirpath, dirname, dirfile in os.walk(imgdir):
        for file in dirfile:
            if file.split('.')[-1] in imgtype:
                t = time.time()
                fp = open(os.path.join(
                    dirpath, file.split('.')[0] + '.txt'), 'w', encoding='utf-8')
                filepath = os.path.join(dirpath, file)
                print(filepath, ':')
                fp.write(filepath + ':\n')
                img = np.array(Image.open(filepath).convert('RGB'))
                result = text_predict(img)
                text = ' '.join([i['text'] for i in result])
                print('text:', text)
                for n, dic in enumerate(result):
                    fp.write('line%d:' % n + str(dic) + '\n')
                timeTake = time.time() - t
                print('recog spend time:%d s' % timeTake)
                fp.write('recog spend time:%d s' % timeTake + '\n')
                fp.close()
    print('total spend time:%d s\n' % (time.time() - t))


if __name__ == '__main__':
    videosDetect()
    # imagesDetect()
