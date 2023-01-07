import cv2
import numpy
import statistics

def contour_to_square(contour):
    ratio_threshold = 0.75
    bb_area_threshold = 0.75
    approx_area_threshold = 0.9
    
    boundingRect = cv2.boundingRect(contour)
    if boundingRect[2]/boundingRect[3] < ratio_threshold or boundingRect[3]/boundingRect[2] < ratio_threshold:
        return None
    area = cv2.contourArea(contour)
    if area < 100:
        return None
    if area / (boundingRect[2] * boundingRect[3]) < bb_area_threshold:
        return None

    approxContour = cv2.approxPolyDP(contour, cv2.arcLength(contour, True)*0.1, True)
    if len(approxContour) != 4:
        return None

    if area / cv2.contourArea(approxContour) < approx_area_threshold:
        return None

    if not cv2.isContourConvex(approxContour):
        return None
 
    return approxContour

def filter_out_wrong_size(contours):
    size_threshold = 0.75

    sizes = []
    for contour in contours:
        if contour is None:
            continue
        bb = cv2.boundingRect(contour)
        sizes.append(bb[2] + bb[3])
    
    if len(sizes) == 0:
        return

    size = statistics.median(sizes)

    for i, contour in enumerate(contours):
        bb = cv2.boundingRect(contour)
        s = bb[2] + bb[3]
        if s / size <= size_threshold or size / s <= size_threshold:
            contours[i] = None


def filter_contours(contours, hierarchy):
    filtered_contours = []
    for i in range(len(contours)):
        if contours[i] is None:
            continue
        parent = hierarchy[0][i][3]
        while parent >= 0 and contours[parent] is None:
            parent = hierarchy[0][parent][3]
        if parent >= 0:
            continue
        filtered_contours.append(contours[i])

    return filtered_contours

def extract(img, contour):
    target_size = 32

    pts1 = numpy.float32([contour[0][0],contour[1][0],contour[2][0],contour[3][0]])
    pts2 = numpy.float32([[-2,-2],[target_size+2,-2],[-2,target_size+2],[target_size+2,target_size+2]])

    def cmp(p1, p2):
        threshold = 10
        
        dx = p1[0] - p2[0]
        dy = p1[1] - p2[1]

        if abs(dy) >= threshold:
            return dy
        if abs(dx) >= threshold:
            return dx
        return 0

    for j in range(len(pts1)):
        for i in range(len(pts1)-1):
            if (cmp(pts1[i], pts1[i+1]) > 0):
                pts1[i], pts1[i+1] = pts1[i+1].copy(), pts1[i].copy()

    M = cv2.getPerspectiveTransform(pts1,pts2)
    return cv2.warpPerspective(img,M,(target_size,target_size))

def empty_process_contours(contours):
    return contours

def main_loop(process, process_contours = empty_process_contours):
    webcam = cv2.VideoCapture(0)
    while True:
        try:
            _, frame = webcam.read()
            framegray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            edges = cv2.Canny(framegray,25,50,1)
            ret, thresh = cv2.threshold(edges, 127, 255, 0)
            contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            contours = list(map(contour_to_square, contours)) 
            filter_out_wrong_size(contours)
            contours = filter_contours(contours, hierarchy)

            processed_contours = process_contours(contours)
        
            display_frame = frame.copy()
            cv2.drawContours(display_frame, contours, -1, (0, 255, 0), 3)
            cv2.imshow("Capturing", display_frame)
            if len(processed_contours) >= 5:        
                extracted_squares = [extract(framegray, contour) for contour in processed_contours]
                process(extracted_squares)
            key = cv2.waitKey(50)
        except(KeyboardInterrupt):
            break

    webcam.release()
    cv2.destroyAllWindows()