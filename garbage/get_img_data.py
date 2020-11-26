import os
import math
import cv2 as cv
import numpy as np
import imgaug.augmenters as iaa


def rotate_about_center(src, angle, scale=1.):
    '''
    旋转缩放处理
    :param src: 图像
    :param angle: 旋转角度
    :param scale: 缩放比例
    :return: 处理后的结果
    '''
    # 获得图像的长和高
    h, w = src.shape[:2]
    # 角度值转弧度值
    rangle = np.deg2rad(angle)
    # 计算旋转缩放后的长和高
    nw = (abs(np.sin(rangle) * h) + abs(np.cos(rangle) * w)) * scale
    nh = (abs(np.cos(rangle) * h) + abs(np.sin(rangle) * w)) * scale
    # 计算2D旋转的仿射矩阵。
    rot_mat = cv.getRotationMatrix2D((nw * 0.5, nh * 0.5), angle, scale)
    # 结合旋转计算从旧中心到新中心的移动
    rot_move = np.dot(rot_mat, np.array([(nw - w) * 0.5, (nh - h) * 0.5, 0]))
    # the move only affects the translation, so update the translation part of the transform
    rot_mat[0, 2] += rot_move[0]
    rot_mat[1, 2] += rot_move[1]
    # 对图像应用仿射变换。
    tempImg = cv.warpAffine(src, rot_mat, (int(math.ceil(nw)), int(math.ceil(nh))), flags=cv.INTER_LANCZOS4, borderValue=(0, 255, 0))
    # 平均取出[0.4,0.8]的20个数据
    seedx = np.linspace(0.4, 0.8, 20)
    seedy = np.linspace(0.4, 0.8, 20)
    # 随机取出这20个数据中的某一个
    indexx = np.random.randint(0, len(seedx))
    indexy = np.random.randint(0, len(seedy))
    # 调整图像大小
    res = cv.resize(tempImg,
                    (int(tempImg.shape[1] * seedx[indexx]), int(tempImg.shape[0] * seedy[indexy])),
                    interpolation=cv.INTER_CUBIC)
    return res


def roate_zoom_img(img):
    # 角度[-30,30]范围内平均取20个数据
    angle = np.linspace(-30, 30, 20)
    # 缩放比例[0.8,1]范围内平均取20个数据
    scale = np.linspace(0.8, 1, 20)
    # 随机取出一个
    index = np.random.randint(0, len(angle))
    # new_img = rotate_about_center(img, 0, scale[index])
    new_img = rotate_about_center(img, angle[index], scale[index])
    return new_img


def img_random_brightness(img):
    '''
    光照增强
    '''
    # 使用imgaug增强器
    brightness = iaa.Multiply((0.9, 1.1))
    # print(brightness.augment)
    image = brightness.augment_image(img)
    return image


def random_augment(img):
    # 同时进行光照旋转缩放增强
    image = img_random_brightness(img)
    image = roate_zoom_img(image)
    return image


def checkOverlap(rectangles, b):
    # 检查是否越界
    for a in rectangles:
        widthmin = min([a[0], a[2], b[0], b[2]])
        widthmax = max([a[0], a[2], b[0], b[2]])
        heightmin = min([a[1], a[3], b[1], b[3]])
        heightmax = max([a[1], a[3], b[1], b[3]])
        if (a[2] - a[0] + b[2] - b[0]) < (widthmax - widthmin) and (
                a[3] - a[1] + b[3] - b[1]) < (heightmax - heightmin):
            continue
        else:
            return True
    return False


# 随机位置生成图片
def transparentOverlay(path):
    # 从文件加载图像
    bgImg = cv.imread(path, -1)
    # 重置图像大小
    target = cv.resize(bgImg, (416, 416))
    rows, cols, _ = target.shape  # 背景图
    rectangles = []
    label = ' '
    for i in range(0, 10):
        index = np.random.randint(0, 16)
        readimg = cv.imread('./image/' + str(index) + '.png')
        readimg = cv.resize(readimg, (200, 200))
        overlay = random_augment(readimg)
        h, w = overlay.shape[:2]
        x = np.random.randint(0, 416 - h)
        y = np.random.randint(0, 416 - w)
        if checkOverlap(rectangles, (y, x, y + w, x + h)):
            continue
        hsv = cv.cvtColor(overlay, cv.COLOR_BGR2HSV)
        lowerb = (35, 43, 36)
        upperb = (77, 255, 255)
        dst = cv.inRange(hsv, lowerb, upperb)
        overlay[dst == 255] = (0, 0, 0)
        color = target[x:x + h, y:y + w]
        color[dst != 255] = (0, 0, 0)
        img = color + overlay
        # 图像合并
        for i in range(img.shape[0]):
            for j in range(img.shape[1]):
                if x + i >= rows or y + j >= cols:
                    continue
                if (hsv[i][j][0:1] > 36 and hsv[i][j][1:2] > 25 and hsv[i][j][2:3] > 25 and hsv[i][j][0:1] < 70):
                    continue
                target[x + i][y + j] = img[i][j][:3]
        rectangles.append((y, x, y + w, x + h))
        # cv.rectangle(target, (y, x), (y + w, x + h), (0, 255, 0), 2)
        # label += "{},{},{},{},{} ".format(y, x, y + w, x + h, index)
        label += "{} {} {} {} {} \n".format(index, y / 416 + w / 832, x / 416 + h / 832, w / 416, h / 416)
    return target, label


# def generateImage():
#     rootdir = './texture'
#     list = os.listdir(rootdir)  # 列出文件夹下所有的目录与文件
#     with open("./train/test.txt", "w") as wf:
#         for i in range(0, 20):
#             index = np.random.randint(0, len(list))
#             path = os.path.join(rootdir, list[index])
#             overlay, label = transparentOverlay(path)
#             # print(label)
#             # cv.imshow('demo', overlay)
#             # cv.waitKey(0)
#             # cv.destroyAllWindows()
#             cv.imwrite("./train/" + str(i) + ".jpg", overlay)
#             annotation = ("./train/" + str(i) + ".jpg" + label)
#             wf.write(annotation + "\n")
#             wf.flush()


def generateImage():
    rootdir = './texture'
    # 列出文件夹下所有的目录与文件
    list = os.listdir(rootdir)
    for i in range(0, 3000):
        index = np.random.randint(0, len(list))
        txt_path = os.path.join(rootdir, list[index])
        overlay, label = transparentOverlay(txt_path)
        cv.imwrite("./train/images/" + str(i) + ".jpg", overlay)
        with open("./train/labels/" + str(i) + ".txt", "w") as wf:
            wf.write(label)
            wf.flush()


if __name__ == '__main__':
    generateImage()
