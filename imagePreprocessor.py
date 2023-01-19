import cv2
import numpy as np
import imutils


def showImg(name, img):
  cv2.imshow(name, img)
  cv2.waitKey(0)


CLASS_NAMES = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H',
               'I', 'J', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']


def findPlate(img):
  img = imutils.resize(img, width=512)

  gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

  bfilter = cv2.bilateralFilter(gray, 11, 17, 17)
  edged = cv2.Canny(bfilter, 30, 220)

  contours, _ = cv2.findContours(
    edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
  contours = sorted(contours, key=cv2.contourArea, reverse=True)

  roi = None
  for contour in contours:
    perimeter = cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, 0.02 * perimeter, True)
    if len(approx) == 4:
      roi = approx
      break

  roi = np.array([roi], np.int32)
  points = roi.reshape(4, 2)
  x, y = np.split(points, [-1], axis=1)

  (x1, x2) = (np.min(x), np.max(x))
  (y1, y2) = (np.min(y), np.max(y))
  number_plate = img[y1:y2, x1:x2]
  number_plate = imutils.resize(number_plate, 256, 512)

  # showImg('original', img)
  # showImg('grayscale', gray)
  # showImg('edges', edged)
  # showImg('plate', number_plate)

  return number_plate


def findBoxes(img):
  im = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
  im = ~im
  binaryIm = cv2.adaptiveThreshold(
    im, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, -1)

  num_of_objects, labeledImage, componentStats, _ = cv2.connectedComponentsWithStats(
    binaryIm)

  remainingComponentLabels = [i for i in range(
    1, num_of_objects) if componentStats[i][4] >= 15]

  filteredImage = np.where(
    np.isin(labeledImage, remainingComponentLabels) == True, 255, 0).astype('uint8')

  contours, hierarchy = cv2.findContours(
    filteredImage, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
  contours_poly = [None] * len(contours)
  boundRect = []

  for i, c in enumerate(contours):
    if hierarchy[0][i][3] == -1:
      contours_poly[i] = cv2.approxPolyDP(c, 3, True)
      if cv2.boundingRect(contours_poly[i])[2] * cv2.boundingRect(contours_poly[i])[3] > 400:
        ratio = cv2.boundingRect(contours_poly[i])[
            2] / cv2.boundingRect(contours_poly[i])[3]
        if ratio < 1 and ratio > .25:
          boundRect.append(cv2.boundingRect(contours_poly[i]))

  temp = np.empty((len(boundRect), 4), dtype=np.int16)

  for i in range(len(boundRect)):
    temp[i][0] = boundRect[i][0]
    temp[i][1] = boundRect[i][1]
    temp[i][2] = boundRect[i][0] + boundRect[i][2]
    temp[i][3] = boundRect[i][1] + boundRect[i][3]

  return temp


def readCharacters(coordinates_array, img, visualize=False):
  import tensorflow as tf
  model = tf.keras.models.load_model('licensePlatesReader.model')
  licensePlate = []

  sorted = coordinates_array[coordinates_array[:, 0].argsort()]

  for i in range(len(sorted)):
    character = img[sorted[i][1]:sorted[i][3],
                    sorted[i][0]:sorted[i][2]]

    tf_image = np.empty((1, 128, 64, 3))
    tf_image[0] = cv2.resize(
      character, (64, 128), interpolation=cv2.INTER_NEAREST)

    if visualize:
      cv2.imshow('character', character)
      cv2.waitKey(0)
      cv2.destroyAllWindows()

    predicted_char = CLASS_NAMES[np.argmax(model.predict([tf_image]))]
    licensePlate.append(predicted_char)

  return licensePlate
