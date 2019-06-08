import numpy as np
import cv2
import random
import sys
from argparse import ArgumentParser
import os
from pydub import AudioSegment
from concurrent.futures import ThreadPoolExecutor
if os.name == 'nt':
    import ctypes


parser = ArgumentParser(description='Encrypt image with code')

group = parser.add_mutually_exclusive_group()
group.add_argument('-e', '--encrypt')
group.add_argument('-d', '--decrypt')

parser.add_argument('-v', '--video', action='store_true')

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


executor = ThreadPoolExecutor(max_workers=8)


def encrypt_video():
    vidcap = cv2.VideoCapture(args.encrypt)
    frame_rate = vidcap.get(cv2.CAP_PROP_FPS)
    success, image = vidcap.read()
    count = 0
    success = True
    if not os.path.exists('./frames/'):
        os.makedirs('./frames/')
    while success:
        success, image = vidcap.read()
        cv2.imwrite("./frames/frame%d.jpg" %
                    count, image)     # save frame as JPEG file
        if cv2.waitKey(10) == 27:                     # exit if Escape is hit
            break
        count += 1

    audio = AudioSegment.from_file(args.encrypt)
    number_files = sum([len(files) for r, d, files in os.walk("./frames/")])
    audioArr = []
    currAudio = 0
    audio_s = (audio.duration_seconds * 1000) / number_files
    while int(currAudio) <= audio.duration_seconds * 1000:
        currAudio += audio_s
        audioArr.append(audio[currAudio-audio_s:currAudio])

    random.seed(args.seed)
    random.shuffle(audioArr)
    shuffledAudio = AudioSegment.empty()
    for audioSegment in audioArr:
        shuffledAudio += audioSegment

    shuffledAudio.export("test.mp3", format='mp3')
    seedLength = 0
    if not os.path.exists('./encryptedFrames/'):
        os.makedirs('./encryptedFrames/')

    for count, frame in enumerate(sorted(os.listdir('./frames/')), start=1):
        executor.submit(encrypt, args.seed+str(seedLength),
                        './frames/'+frame, './encryptedFrames/'+frame)

        seedLength += 1

    executor.shutdown(wait=True)


def encrypt(seed, file_name, save_name):
    print(file_name)
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
    print("Shuffled with seed {}".format(seed))
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
        cv2.imwrite(save_name, blank_image)
        print("Saved as {}".format(save_name))
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
    if args.video:
        encrypt_video()
    else:
        if args.encrypt:
            img = encrypt(seed, args.encrypt)
            show_image(img)
        if args.decrypt:
            img = decrypt(seed, args.decrypt)
            show_image(img)
