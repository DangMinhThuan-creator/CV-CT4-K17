import numpy as np
import cv2 as cv
import urllib.request

def read_img_from_url(url):
    req = urllib.request.urlopen(url)
    img_rw = np.asarray(bytearray(req.read()), dtype="uint8")
    img = cv.imdecode(img_rw, 3)
    return img


if __name__ == "__main__":
    url = "https://raw.githubusercontent.com/opencv/opencv/refs/heads/4.x/samples/data/lena.jpg"
    anh_goc =read_img_from_url(url)
    anh_nam = add_noise(anh_goc)
    anh_muoi_tieu = add_muoi_tieu(anh_goc,0.03)
    img2 = anh_muoi_tieu.copy()
    clean_img = cv.blur(img2, (3,3))
    img3 = np.concatenate((anh_muoi_tieu, clean_img), axis=1)
    cv.imshow("img3", img3)
    cv.waitKey(0)
    cv.destroyAllWindows()
    img5 = anh_muoi_tieu.copy()
    clean_img = cv.medianBlur(img5, 3)
    img6 = np.concatenate((anh_muoi_tieu, clean_img, anh_goc), axis=1)
    cv.imshow("img6", img6)
    cv.waitKey(0)
    cv.destroyAllWindows()

    ed1 = cv.Canny(anh_muoi_tieu, 50, 150)
    ed2 = cv.Canny(clean_img, 50, 150)
    ed3 = cv.Canny(anh_goc, 50, 150)
    img7 = np.concatenate((ed1, ed2, ed3), axis=1)
    cv.imshow("img7", img7)
    cv.waitKey(0)
    cv.destroyAllWindows()
    h, w = edge.shape
    polygon = np.array([[0, h], [w // 2, h // 2], [w, h]])