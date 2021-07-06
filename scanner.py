import os.path
import sys
import cv2
import argparse
import fpdf
import numpy as np
import tempfile


def parse_arguments(args=None):
    parser = argparse.ArgumentParser(description="Random description", allow_abbrev=True)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("-p", "--pdf", nargs=1)
    group.add_argument("-v", "--view", action="store_true")
    group.add_argument("-i", "--image", nargs=1)
    parser.add_argument("-t", "--target", type=int, nargs=1)
    parser.add_argument("-d", "--dimensions", type=int, nargs=2)
    parser.add_argument("-s", "--scanner", type=int, nargs=2)
    parser.add_argument("filenames", nargs="+")
    return parser.parse_args(args)


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
    document_approximation = None
    for contour in contours:
        perimeter = cv2.arcLength(contour, True)
        approximation = cv2.approxPolyDP(contour, 0.01 * perimeter, True)

        if len(approximation) == 4:
            document_approximation = approximation
            document_contour = contour
            break

    return document_approximation, document_contour


def transform_to_birds_eye(img, document_approximation):
    corner_points = document_approximation.reshape(4, 2).astype("float32")
    corner_points = sorted(corner_points, key=lambda x: x[0])
    left = sorted(corner_points[:2], key=lambda x: x[1])
    right = sorted(corner_points[2:], key=lambda x: x[1], reverse=True)
    corner_points = np.asarray(left + right)
    tl, bl, br, tr = corner_points
    width = int(max(np.linalg.norm(tl - tr), np.linalg.norm(bl - br)))
    height = int(max(np.linalg.norm(tl - bl), np.linalg.norm(tr - br)))
    target_points = np.array(
        [
            [0, 0],
            [0, height],
            [width, height],
            [width, 0]
        ], dtype="float32"
    )
    transformation_matrix = cv2.getPerspectiveTransform(corner_points, target_points)

    return cv2.warpPerspective(img, transformation_matrix, (width, height))


def correct_img(img):
    grayscale = convert_to_grayscale_img(img)
    ratio = compute_resize_ratio(grayscale, 500, "height")
    grayscale = resize_img(grayscale, 1 / ratio)
    grayscale = blur_img(grayscale)
    edges = detect_edges(grayscale)
    document_approximation, document_contour = detect_document(edges)
    if document_approximation is None:
        return None
    # TODO: use non-linear transformation for document correction
    # leaving the detailed contour so that maybe I can apply non-linear transformation to correct the resulting image
    # https://mathematica.stackexchange.com/questions/5676/how-to-peel-the-labels-from-marmalade-jars-using-mathematica
    # https://dsp.stackexchange.com/questions/2406/how-to-flatten-the-image-of-a-label-on-a-food-jar
    img = transform_to_birds_eye(img, document_approximation * ratio)

    return img


def scannify(img, block_size, c):
    if block_size < 0:
        error(f"Specified block size {block_size} is less than zero.")
    if c < 0:
        error(f"Specified c {c} is less than zero.")
    img = convert_to_grayscale_img(img)
    if (block_size != 0) and (c != 0):
        if block_size % 2 != 1:
            error(f"Specified block size {block_size} isn't odd.")
        img = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, block_size, c)
    return img


def show_img(img, window_name, scale=1):
    cv2.imshow(window_name, resize_img(img, scale))
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def validate_filename(filename, extension):
    return filename.split(".")[1] == extension


def error(msg):
    print("Error:", msg, "Aborting process.", file=sys.stderr)
    sys.exit(1)


def view(filenames, out_images, args):
    if args.target:
        height = args.target[0]
    else:
        height = 800

    for i, (filename, img) in enumerate(zip(filenames, out_images)):
        ratio = compute_resize_ratio(img, height)
        show_img(img, f"{filename} ({i + 1})", 1 / ratio)


def pdf(_filenames, out_images, args):
    pdf_filename = args.pdf[0]
    if not validate_filename(pdf_filename, "pdf"):
        error(f"Name {pdf_filename} isn't valid name for .pdf file.")

    if args.dimensions:
        w, h = args.dimensions
    else:
        w = 210
        h = 297

    with tempfile.TemporaryDirectory() as temp_dir:
        pdf_file = fpdf.FPDF()
        pdf_file.compress = False
        for i, img in enumerate(out_images):
            img_filename = temp_dir + str(i) + ".png"
            cv2.imwrite(img_filename, img)
            pdf_file.add_page()
            pdf_file.image(img_filename, x=0, y=0, w=w, h=h)

        pdf_file.output(pdf_filename, "F")


def image(filenames, out_images, args):
    out_dir = args.image[0]
    if not os.path.isdir(out_dir):
        error(f"Directory {out_dir} doesn't exist.")

    for filename, img in zip(filenames, out_images):
        cv2.imwrite(out_dir + "/" + os.path.basename(os.path.normpath(filename)), img)


def main(sample_arguments=None):
    args = parse_arguments(sample_arguments)

    filenames = args.filenames

    out_images = []
    for i, filename in enumerate(filenames):
        img = cv2.imread(filename, cv2.IMREAD_COLOR)
        if img is None:
            error(f"Image {filename} ({i + 1}) doesn't exist.")
        out_img = correct_img(img)
        if out_img is None:
            error(f"Couldn't detect document in image {filename} ({i + 1}).")
        if args.scanner:
            out_img = scannify(out_img, args.scanner[0], args.scanner[1])
        out_images.append(out_img)

    if args.view:
        view(filenames, out_images, args)
    elif args.pdf:
        pdf(filenames, out_images, args)
    elif args.image:
        image(filenames, out_images, args)


if __name__ == '__main__':
    main()
