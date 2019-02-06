import numpy as np
import cv2
import random
import sys
from argparse import ArgumentParser

parser = ArgumentParser(description='Encrypt image with code')

group = parser.add_mutually_exclusive_group()
group.add_argument('-e', '--encrypt')
group.add_argument('-d', '--decrypt')

parser.add_argument('-o', '--output', type=str,
                    help='Output file name')
parser.add_argument('-f', '--factor', type=int, default=1, help='Scale factor')
parser.add_argument('-c', '--color', type=int, default=-
                    1, help='<0 RGBA, 0 greyscale, >0 RGB')

parser.add_argument('-s', '--seed', type=str, help='Seed', required=True)

args = parser.parse_args()

if (args.encrypt or args.decrypt) is None:
    raise ValueError('Specify file name or path to file')


def show_image(img, scale=args.factor):
    height, width = img.shape[:2]
    cv2.namedWindow('image', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('image', int(width / scale), int(height / scale))
    cv2.moveWindow('image', 100, 100)
    cv2.imshow('image', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def encrypt(seed, file_name, save_name=args.output):
    img_show = cv2.imread(file_name, args.color)

    height, width = img_show.shape[:2]
    blank_image = np.zeros(img_show.shape, np.uint8)

    print("{} - Height : {}, Width : {}".format(
        args.decrypt, height, width))

    myPixels = []

    for x in range(height):
        for y in range(width):
            myPixels.append(img_show[x, y])
    print("Pixels loaded into list;")
    random.seed(seed)
    random.shuffle(myPixels)
    print("Shuffled with seed {}".format(args.seed))
    tmp = 0
    for x in range(height):
        for y in range(width):
            blank_image[x, y] = myPixels[tmp]
            tmp += 1
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

    myPixels = []

    for x in range(height):
        for y in range(width):
            myPixels.append(img_show[x, y])
    print("Pixels loaded into list;")

    tmp = 0
    Decrypted = list(range(len(myPixels)))
    random.seed(seed)
    random.shuffle(Decrypted)

    originalList = [0]*len(myPixels)   # empty list, but the right length
    for index, originalIndex in enumerate(Decrypted):
        originalList[originalIndex] = myPixels[index]

    for x in range(height):
        for y in range(width):
            blank_image[x, y] = originalList[tmp]
            tmp += 1
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
