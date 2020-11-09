import numpy as np
import cv2
import torch


class ImageTransformer:
    def __init__(self, debug=False):
        self.kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        self.debug = debug
        self.debug_img = None

    def transform(self, *imgs, inversion=False):
        """ Transform N bgr-images to grayscale [N, 1, 28, 28] float pytorch tensor """
        if not imgs:
            return None

        if inversion:
            thresh_mode = cv2.THRESH_BINARY_INV
            border_value = 0
        else:
            thresh_mode = cv2.THRESH_BINARY
            border_value = 255

        transformed_list = []
        for img in imgs:
            img_transformed = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            img_transformed = cv2.adaptiveThreshold(img_transformed, 255,
                                                    cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                                    thresh_mode, 21, 5)

            img_transformed = cv2.morphologyEx(img_transformed, cv2.MORPH_OPEN, self.kernel, iterations=1)
            img_transformed = cv2.morphologyEx(img_transformed, cv2.MORPH_CLOSE, self.kernel, iterations=2)
            img_transformed = cv2.GaussianBlur(img_transformed, (7, 7), 1)

            # Add border to make it look more like MNIST
            br = img_transformed.shape[1] // 3
            img_transformed = cv2.copyMakeBorder(img_transformed, br, br, br, br,
                                                 cv2.BORDER_CONSTANT, value=border_value)
            # Resize to 28x28 pixels
            img_transformed = cv2.resize(img_transformed, (28, 28), interpolation=cv2.INTER_AREA)
            img_transformed = np.array(img_transformed, dtype=np.single) / 255
            transformed_list.append(img_transformed)

        if self.debug:
            self.debug_img = np.concatenate(transformed_list, axis=1)
        return torch.tensor(transformed_list).view(-1, 1, 28, 28)
