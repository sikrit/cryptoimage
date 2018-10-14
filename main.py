import numpy as np
import cv2
import random

seed = 'rng3'
img = 'image.jpg'
scale_ratio = 2

img_show = cv2.imread(img,3)
height, width = img_show.shape[:2]

height = height - 1
width = width - 1

print("Height : {}, Width : {}, Type : {} {}".format(height, width, type(height), type(width)))

random.seed(seed)

def show_image():

    cv2.namedWindow('image',cv2.WINDOW_NORMAL)
    cv2.resizeWindow('image', width, height)

    cv2.imshow('image',img_show)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
if __name__ == '__main__':
        
    randX = []
    randY = []
    
    for x in range(height):
        for y in range(width):
            print("{} {}".format(x, y))
            pixel = img_show[x, y]
            rand_x = random.randint(0, height-1)
            rand_y = random.randint(0, width-1)
            pixel_random = img_show[rand_x, rand_y]
            img_show[x,y]=pixel_random
            img_show[rand_x,rand_y]=pixel
            
    for x in range(height):
        for y in range(width):
            randX.append(random.randint(0, height-1))
            randY.append(random.randint(0, width-1))

    for x in range(height, -1, -1):
        for y in range(width, -1, -1):
            print("{} {}".format(x, y))
            pixel = img_show[x, y]
            rand_x = randX[::-1][x]
            rand_y = randY[::-1][y]
            pixel_random = img_show[rand_x, rand_y]
            img_show[x,y]=pixel_random
            img_show[rand_x,rand_y]=pixel

    show_image()


    cv2.destroyAllWindows()