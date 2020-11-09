import cv2
import numpy as np


class DigitFounder:
    def __init__(self, debug=False):
        self.kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        self.debug = debug
        self.debug_img = np.zeros((10, 10))

    def get_positions(self, img, thd1=30, thd2=200, area_min=50, area_max=10000) -> list:
        """Find rectangles around contours. Return list of tuples like (x, y, w, h)"""

        # switch to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # use Canny edge detector
        img_canny = cv2.Canny(gray, thd1, thd2)
        # use Close operation to close contours after canny
        img_closed = cv2.morphologyEx(
            img_canny,
            cv2.MORPH_CLOSE,
            self.kernel,
            iterations=5
        )

        # For debug
        if self.debug:
            self.debug_img = img_closed.copy()

        coordinate_list = []

        cnts, hierarchy = cv2.findContours(img_closed, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        if cnts:
            hierarchy = hierarchy[0]  # get the actual inner list of hierarchy descriptions

            # hierarchy: [[next_cnt, prev_cnt, child, parent], ...]
            # Grab only outer contours

            outer_contours = [cnt[0] for cnt in zip(cnts, hierarchy)
                              if cnt[1][3] < 0 and area_min < cv2.contourArea(cnt[0]) < area_max]

            for cnt in outer_contours:
                x, y, w, h = cv2.boundingRect(cnt)

                # transform rectangle to square to minimize interpolation distortion
                if h > w:
                    x = max(x - (h - w) // 2, 0)
                    w = min(h, img.shape[1] - x)
                else:
                    y = max(y - (w - h) // 2, 0)
                    h = min(w, img.shape[0] - y)

                coordinate_list.append((x, y, w, h))
        return coordinate_list
