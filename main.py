import cv2
import argparse
import numpy as np
import math
import itertools
from fpdf import FPDF


def parse_arguments():
    parser = argparse.ArgumentParser(description="Random description", allow_abbrev=True)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("-p", "--pdf", nargs=1)
    group.add_argument("-s", "--show", action="store_true")
    parser.add_argument("filenames", nargs="+")
    return parser.parse_args(["-s", "data/1.jpg"])


def convert_to_grayscale_img(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def compute_resize_ratio(img, target, target_dimension="height"):
    shape = img.shape
    dimension = 0
    if target_dimension == "height":
        dimension = shape[0]
    elif target_dimension == "width":
        dimension = shape[1]
    return dimension / target


def resize_img(img, scale, interpolation=cv2.INTER_AREA):
    return cv2.resize(img, (0, 0), fx=scale, fy=scale, interpolation=interpolation)


def blur_img(img, size=(5, 5), sigma=1):
    return cv2.GaussianBlur(img, size, sigma)


def detect_edges(img, threshold1=50, threshold2=200):
    return cv2.Canny(img, threshold1, threshold2)


def detect_document(img):
    contours, hierarchy = cv2.findContours(img, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    document_contour = None
    for contour in contours:
        perimeter = cv2.arcLength(contour, True)
        approximation = cv2.approxPolyDP(contour, 0.01 * perimeter, True)

        if len(approximation) == 4:
            document_contour = contour
            break

    if document_contour is not None:
        disp_img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        cv2.drawContours(disp_img, [document_contour], -1, (0, 0, 255))
        cv2.imshow("Contours", disp_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


def correct_img(img):
    grayscale = convert_to_grayscale_img(img)
    ratio = compute_resize_ratio(grayscale, 500, "height")
    grayscale = resize_img(grayscale, 1 / ratio)
    grayscale = blur_img(grayscale)
    edges = detect_edges(grayscale)
    cv2.imshow("Original (resized)", resize_img(img, 1 / ratio))
    cv2.imshow("Edges", edges)
    cv2.waitKey(0)
    detect_document(edges)

    return img


def main():
    args = parse_arguments()

    filenames = args.filenames

    for filename in filenames:
        img = cv2.imread(filename, cv2.IMREAD_COLOR)
        correct_img(img)


if __name__ == '__main__':
    main()

