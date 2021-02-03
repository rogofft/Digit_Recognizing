# This is a pet-project for recognizing handwritten digits on video stream.
# Used PyTorch, OpenCV, NumPy, etc...

import cv2
import torch
import os
import sys

# Custom modules
import camstream
import image_correction as imcor
import image_transformer
import digit_founder
from model import get_model, device

# parsing arguments for --debug
if any(arg == '--debug' for arg in sys.argv):
    Debug = True
else:
    Debug = False

# Neural Network Setup
print('Using ' + str(device))
net = get_model()
# Load weights from fitted net
net_path = 'model/model_config.ptn'
if os.path.isfile(net_path):
    net.load_state_dict(torch.load(net_path))
else:
    print('Neural Network not fitted!')
net.eval()

# Create windows
cv2.namedWindow('origin', cv2.WINDOW_NORMAL)
if Debug:
    cv2.namedWindow('debug_Corrected', cv2.WINDOW_NORMAL)
    cv2.namedWindow('debug_DigitFounder', cv2.WINDOW_NORMAL)
    cv2.namedWindow('debug_Transformer', cv2.WINDOW_AUTOSIZE)

# Video capture
cam = camstream.CamStream(src=0).start()
# Class for finding contours
founder = digit_founder.DigitFounder(debug=Debug)
# Class for transform images to neural network format
transformer = image_transformer.ImageTransformer(debug=Debug)

# while window not closed, or not pressed 'q'
while cv2.getWindowProperty('origin', 0) >= 0:
    # Reading from camera
    _, img = cam.read()

    # Choose image correction method:

    # corrected_img = img.copy()
    corrected_img = imcor.gray_world_bgr(img)
    # corrected_img = imcor.linear_correction_bgr(img)
    # corrected_img = imcor.gamma_corr(img, 2.2)
    # corrected_img = imcor.retinex(img, 50.)

    corrected_img = cv2.medianBlur(corrected_img, 5)

    # Take coordinates of digits on image
    digit_coords = founder.get_positions(corrected_img)
    # Take images of digits from coordinates
    digits = []
    for x, y, w, h in digit_coords:
        digits.append(corrected_img[y:y + h, x:x + w])

    # Transform digit images to neural network format
    digit_tensor = transformer.transform(*digits, inversion=True)
    # Check digit_tensor not None
    if digit_tensor is not None:
        # use CNN to predict digit
        with torch.no_grad():
            predictions = net.predict(digit_tensor.to(device))

        # visualize all what we doing
        for ((x, y, w, h), prediction) in zip(digit_coords, predictions):
            cv2.rectangle(img, (x, y), (x + w, y + h), color=[0, 255, 0])
            cv2.putText(img, f'{prediction.item()}', (x, y+h+20),
                        cv2.FONT_HERSHEY_DUPLEX, 0.5, (0, 255, 255), 1)

    # update all windows
    cv2.imshow('origin', img)
    if Debug:
        cv2.imshow('debug_Corrected', corrected_img)
        cv2.imshow('debug_DigitFounder', founder.debug_img)
        cv2.imshow('debug_Transformer', transformer.debug_img)

    # Wait 'q' for close application
    if cv2.waitKey(16) & 0xFF == ord('q'):
        break

cam.stop()
del cam
cv2.destroyAllWindows()
