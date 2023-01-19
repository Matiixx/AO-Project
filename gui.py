import cv2
import numpy as np
import PySimpleGUI as sg
from imagePreprocessor import findPlate, findBoxes, readCharacters
from platesbuttons import plate1_base64, plate2_base64


def get_image(path):
  img = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
  img = cv2.resize(img, (850, 600))
  imgbytes = cv2.imencode('.ppm', img)[1].tobytes()
  return imgbytes


def createGUI():
  sg.theme('DarkAmber')

  layout_v = [
      [sg.Text("Wybierz zdjęcie: ", pad=(0, 10)), sg.Input(),
       sg.FileBrowse(key="-IN-", button_text='Znajdź plik', ), ],
      [sg.Button(".", image_data=plate1_base64,
                 button_color=(sg.theme_background_color(), sg.theme_background_color()), border_width=0,)],
      [sg.Text("", key='-OUTPUT-', pad=(0, 10))]
  ]

  layout = [[sg.VPush()],
            [sg.Push(), sg.Column(layout_v, element_justification='c'), sg.Push()],
            [sg.VPush()]]

  window = sg.Window('Rozpoznawanie tablic rejestracyjnych',
                     layout, size=(800, 600), finalize=True)
  result = ''
  path = ''

  while True:
    event, values = window.read()

    if event == sg.WIN_CLOSED or event == ',':
      break

    if event == ".":
      path = values["-IN-"]
      img = cv2.imread(path)
      licensePlate = findPlate(img)
      boundingBoxes = findBoxes(licensePlate)
      plate = readCharacters(boundingBoxes, licensePlate)
      for char in plate:
        result += str(char)
      break

    if event == "Live View":
      break

  window.close()
  end_layout = [
      [sg.Text(f"Tablica rejestracyjna: {result}", size=(
        len(result) + 50, 1), pad=(250, 10), justification='center')],
      [sg.Image(get_image(path), size=(850, 600))],
      [sg.Column([[sg.Button(",", image_data=plate2_base64,
                             button_color=(sg.theme_background_color(), sg.theme_background_color()), border_width=0,)],], element_justification='center', expand_x=True)],
  ]

  end_window = sg.Window("Rozpoznana tablica", end_layout,
                         size=(900, 730), finalize=True)
  while True:
    event, _ = end_window.read()
    if event == sg.WIN_CLOSED or event == ',':
      break
