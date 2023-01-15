import cv2
import imageio
import itertools
import numpy
import statistics
from mss import mss

def contour_to_square(contour):
    ratio_threshold = 0.5
    bb_area_threshold = 0.6
    approx_area_threshold = 0.8
    approx_perimeter_threshold = 0.8
    max_size = 200
    
    boundingRect = cv2.boundingRect(contour)
    if boundingRect[2]/boundingRect[3] < ratio_threshold or boundingRect[3]/boundingRect[2] < ratio_threshold:
        return None
    area = cv2.contourArea(contour)
    perimeter = cv2.arcLength(contour, True)
    if area < 100:
        return None
    if area / (boundingRect[2] * boundingRect[3]) < bb_area_threshold:
        return None
    if boundingRect[2] > max_size or boundingRect[3] > max_size:
        return None

    approx_factor_left = 0
    approx_factor_right = 0.2
    approx_contour = contour
    for _ in range(10):
        approx_factor = (approx_factor_left + approx_factor_right)/2
        new_approx_contour = cv2.approxPolyDP(contour, cv2.arcLength(contour, True)*approx_factor, True)
        if len(new_approx_contour) < 4:
            approx_factor_right = approx_factor
        else:
            approx_factor_left = approx_factor
        if len(new_approx_contour) == 4:
            approx_contour = new_approx_contour
    
    if len(approx_contour) != 4:
        return None

    if not cv2.isContourConvex(approx_contour):
        return None

    if area / cv2.contourArea(approx_contour) < approx_area_threshold or cv2.contourArea(approx_contour) / area < approx_area_threshold:
        return None

    if perimeter / cv2.arcLength(approx_contour, True) < approx_perimeter_threshold or cv2.arcLength(approx_contour, True) / perimeter < approx_perimeter_threshold:
        return None
 
    return approx_contour

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


def filter_out_child_contours(contours, hierarchy):
    filtered_contours = []
    for i in range(len(contours)):
        if contours[i] is None:
            continue
        parent = hierarchy[0][i][3]
        while parent >= 0 and contours[parent] is None:
            parent = hierarchy[0][parent][3]
        if parent >= 0:
            contours[i] = None

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

def get_redness(img):
    count = 0
    sum = 0
    for x in range(32):
        for y in range(32):
            if img[x][y][0] > 100:
                continue
            count += 1
            sum += (img[x][y][2]/max(img[x][y][1], 1))
    if count == 0:
        return 0
    return sum / count

def empty_process_contours(contours):
    return contours

use_webcam = False

if use_webcam:
    webcam = cv2.VideoCapture(0)
    def get_next_frame():
        _, frame = webcam.read()
        return frame
else:
    sct = mss()
    bounding_box = {'top': 500, 'left': 1500, 'width': 500, 'height': 400}
    def get_next_frame():
        return numpy.array(sct.grab(bounding_box))

def main_loop(process, process_contours = empty_process_contours):
    
    while True:
        try:
            frame = get_next_frame()
            frameblue = frame.copy()
            frameblue[:,:,1] = frameblue[:,:,0]
            frameblue[:,:,2] = frameblue[:,:,0]
            framenoblue = frame.copy()
            framenoblue[:,:,0] = numpy.zeros([framenoblue.shape[0], framenoblue.shape[1]])
            framegray = cv2.cvtColor(frameblue, cv2.COLOR_BGR2GRAY)
            processed_frame = cv2.multiply(framenoblue, (1, 1, 1, 1), scale = 2)
            b, g, r = cv2.split(processed_frame)[0:3]
            processed_frame[:,:,0] = cv2.max(g, r)
            processed_frame[:,:,1] = cv2.max(g, r)
            processed_frame[:,:,2] = cv2.max(g, r)
            processed_frame = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2GRAY)
            thresh = cv2.adaptiveThreshold(processed_frame,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY_INV,9,2)
            contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            contours = list(map(contour_to_square, contours)) 
            filter_out_child_contours(contours, hierarchy)
            filter_out_wrong_size(contours)

            contours = list(filter(lambda c : c is not None, contours))

            processed_contours = process_contours(contours)
        
            display_frame = frame.copy()
            cv2.drawContours(display_frame, contours, -1, (0, 255, 0), 2)
            cv2.imshow("Capturing", display_frame)
            cv2.waitKey(1)
            highlight_contours_ids = []
            if len(processed_contours) >= 5:        
                extracted_squares = [extract(framegray, contour) for contour in processed_contours]
                extracted_squares_redness = [get_redness(extract(frame, contour)) for contour in processed_contours]
                highlight_contours_ids = process(extracted_squares, extracted_squares_redness)

            if highlight_contours_ids is not None:
                highlight_contours = []
                for id in highlight_contours_ids:
                    highlight_contours.append(contours[id])
                display_frame_copy = display_frame.copy()
                cv2.drawContours(display_frame_copy, highlight_contours, -1, (0, 255, 0), 4)
                cv2.imshow("Capturing", display_frame_copy)

            key = cv2.waitKey(50)
            if key == ord('p') or key == ord('g'):
                generate_gif = key == ord('g')
                gif_frames = []
                for time in itertools.count(0):
                    if highlight_contours_ids is not None:
                        display_frame_copy = frame.copy()
                        display_frame_copy = cv2.blur(display_frame_copy, (20, 20), 50)

                        mask = numpy.zeros([frame.shape[0], frame.shape[1]])
                        cv2.drawContours(mask, contours, -1, (255), -1)
                        copy_locs = numpy.where(mask != 0)
                        display_frame_copy[copy_locs[0], copy_locs[1]] = frame[copy_locs[0], copy_locs[1]]

                        contour_time = 10
                        max_highlight = 8

                        cur_contour = time // contour_time % len(highlight_contours)
                        next_contour = (cur_contour + 1) % len(highlight_contours)
                        cur_contour_highlight = max_highlight - int((time % contour_time) / contour_time * max_highlight)
                        next_contour_highlight = max_highlight - cur_contour_highlight

                        cv2.drawContours(display_frame_copy, [highlight_contours[cur_contour]], -1, (0, 255, 0), 1 + cur_contour_highlight)
                        cv2.drawContours(display_frame_copy, [highlight_contours[next_contour]], -1, (0, 255, 0), 1 + next_contour_highlight)
                        cv2.imshow("Capturing", display_frame_copy)
                        if generate_gif and time % 4 == 0:
                            gif_frames.append(cv2.cvtColor(display_frame_copy, cv2.COLOR_BGR2RGB))

                    key = cv2.waitKey(50)
                    if key == ord('p'):
                        break
                if len(gif_frames) > 0:
                    with imageio.get_writer("capture.gif", mode="I") as writer:
                        for frame in gif_frames:
                            writer.append_data(frame)
        except(KeyboardInterrupt):
            break

    if use_webcam:
        webcam.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    def empty_process(extracted_squares, extracted_squares_redness):
        pass

    main_loop(empty_process)