from cv2 import imread
from cv2 import imshow
from cv2 import waitKey
from cv2 import destroyAllWindows
from cv2 import CascadeClassifier
from cv2 import rectangle

img = imread('./pics/test1.jpg')

haarcade = CascadeClassifier(r'C:\Users\rajki\Desktop\main project\data\haarcascade_frontalface_default.xml')
bbox = haarcade.detectMultiScale(img)

for box in bbox:
    x, y, width, height = box
    x2, y2 = x + width, y + height
    rectangle(img, (x,y), (x2,y2), (0,0,255), 1)
    # print(box)
imshow(f'Detected Faces in {img}', img)

waitKey(0)

destroyAllWindows()

