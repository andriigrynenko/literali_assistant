import cv2
import math
import os
import random
import shutil

import webcam_utils

def sort_contours(contours):
    if len(contours) != 25:
        return []

    contours = contours.copy()

    def cmp(c1, c2):
        threshold = 20
        bb1 = cv2.boundingRect(c1)
        bb2 = cv2.boundingRect(c2)

        dx = bb1[0] - bb2[0]
        dy = bb1[1] - bb2[1]

        if abs(dy) >= threshold:
            return dy
        if abs(dx) >= threshold:
            return dx
        return 0

    for j in range(len(contours)):
        for i in range(len(contours)-1):
            if (cmp(contours[i], contours[i+1]) > 0):
                contours[i], contours[i+1] = contours[i+1].copy(), contours[i].copy()

    for i in range(5):
        for j in range(4):
            bb1 = cv2.boundingRect(contours[i*5+j])
            bb2 = cv2.boundingRect(contours[i*5+j+1])
            d = math.dist([bb1[0], bb1[1]], [bb2[0], bb2[1]])
            if d / ((bb1[2]+bb1[3]) / 2) > 1.25:
                return []
        if i == 4:
            continue
        bb1 = cv2.boundingRect(contours[i*5])
        bb2 = cv2.boundingRect(contours[i*5+5])
        d = math.dist([bb1[0], bb1[1]], [bb2[0], bb2[1]])
        if d / ((bb1[2]+bb1[3]) / 2) > 1.25:
            return []

    return contours

output_dir = "./sample_generator_output"

try:
    shutil.rmtree(output_dir)
except OSError as error:
    print(error)   

os.mkdir(output_dir)

for i in range(25):
    os.mkdir(output_dir + '/' + str(i))

def process(extracted_squares, extracted_squares_redness):
    for i, image in enumerate(extracted_squares):
        write = cv2.imwrite(filename=output_dir + '/' + str(i)+'/' + str(random.randint(1000000, 9999999)) +'.png', img=image)
        if not write:
            print("Saving image failed")

webcam_utils.main_loop(process, process_contours = sort_contours)