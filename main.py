import numpy as np
import cv2
import random
import sys
from argparse import ArgumentParser
import os

if os.name == 'nt':
    import ctypes


parser = ArgumentParser(description='Encrypt image with code')

group = parser.add_mutually_exclusive_group()
group.add_argument('-e', '--encrypt')
group.add_argument('-d', '--decrypt')

parser.add_argument('-o', '--output', type=str,
                    help='Output file name')
parser.add_argument('-f', '--factor', type=int, default=1, help='Scale factor')
parser.add_argument('-c', '--color', type=int, default=-
                    1, help='<0 RGBA, 0 greyscale, >0 RGB')
parser.add_argument('-p', '--pixel', type=int,
                    default=1, help='Pixel block size')

parser.add_argument('-s', '--seed', type=str, help='Seed', required=True)

args = parser.parse_args()

if (args.encrypt or args.decrypt) is None:
    raise ValueError('Specify file name or path to file')


def show_image(img, scale=args.factor):
    height, width = img.shape[:2]
    cv2.namedWindow('image', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('image', int(width / scale), int(height / scale))
    if os.name == 'nt':
        cv2.moveWindow('image', int(ctypes.windll.user32.GetSystemMetrics(
            0)/2)-int((width / scale)/2), int(ctypes.windll.user32.GetSystemMetrics(1)/2) - int((height / scale)/2))
    else:
        cv2.moveWindow('image', 400, 100)
    cv2.imshow('image', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def encrypt(seed, file_name, save_name=args.output):
    img_show = cv2.imread(file_name, args.color)

    height, width = img_show.shape[:2]
    blank_image = np.zeros(img_show.shape, np.uint8)

    print("{} - Height : {}, Width : {}".format(
        args.decrypt, height, width))

    myBlocks = []
    for blockX in range(int(width / args.pixel)):
        for blockY in range(int(height / args.pixel)):
            tmp = []
            for x in range(blockX*args.pixel, (blockX+1)*args.pixel):
                for y in range(args.pixel*blockY, args.pixel*(blockY+1)):
                    tmp.append(img_show[y, x])
            myBlocks.append(tmp)
    print("Pixels loaded into list;")
    random.seed(seed)
    random.shuffle(myBlocks)
    print("Shuffled with seed {}".format(args.seed))
    blockId = -1
    pixelId = 0
    for blockX in range(int(width / args.pixel)):
        for blockY in range(int(height / args.pixel)):
            blockId += 1
            pixelId = 0
            for x in range(args.pixel*blockX, args.pixel*(blockX+1)):
                for y in range(args.pixel*blockY, args.pixel*(blockY+1)):
                    blank_image[y, x] = myBlocks[blockId][pixelId]
                    pixelId += 1
    if args.output:
        cv2.imwrite(save_name, blank_image, [
                    cv2.IMWRITE_PNG_COMPRESSION, 0, cv2.IMWRITE_JPEG_QUALITY, 100])
        print("Saved as {}".format(args.output))
    return blank_image


def decrypt(seed, file_name, save_name=args.output):
    img_show = cv2.imread(file_name, args.color)
    height, width = img_show.shape[:2]

    blank_image = np.zeros(img_show.shape, np.uint8)

    print("{} - Height : {}, Width : {}".format(
        args.decrypt, height, width))

    myBlocks = []
    for blockX in range(int(width / args.pixel)):
        for blockY in range(int(height / args.pixel)):
            tmp = []
            for x in range(blockX*args.pixel, (blockX+1)*args.pixel):
                for y in range(args.pixel*blockY, args.pixel*(blockY+1)):
                    tmp.append(img_show[y, x])
            myBlocks.append(tmp)
    print("Pixels loaded into list;")

    Decrypted = list(range(len(myBlocks)))
    random.seed(seed)
    random.shuffle(Decrypted)

    originalList = [0]*len(myBlocks)   # empty list, but the right length
    for index, originalIndex in enumerate(Decrypted):
        originalList[originalIndex] = myBlocks[index]

    blockId = -1
    pixelId = 0
    for blockX in range(int(width / args.pixel)):
        for blockY in range(int(height / args.pixel)):
            blockId += 1
            pixelId = 0
            for x in range(args.pixel*blockX, args.pixel*(blockX+1)):
                for y in range(args.pixel*blockY, args.pixel*(blockY+1)):
                    blank_image[y, x] = originalList[blockId][pixelId]
                    pixelId += 1
    print("Restored original list of pixels with seed {}".format(args.seed))
    if args.output:
        cv2.imwrite(save_name, blank_image, [
                    cv2.IMWRITE_PNG_COMPRESSION, 0, cv2.IMWRITE_JPEG_QUALITY, 100])
        print("Saved as {}".format(args.output))

    return blank_image


if __name__ == '__main__':
    seed = args.seed
    if args.encrypt:
        img = encrypt(seed, args.encrypt)
        show_image(img)
    if args.decrypt:
        img = decrypt(seed, args.decrypt)
        show_image(img)
