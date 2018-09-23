import cv2
from darkflow.net.build import TFNet
import matplotlib.pyplot as plt

options = {
    'model': 'cfg/yolo.cfg',
    'load': 'bin/yolo.weights',
    'threshold': 0.3,
}

tfnet = TFNet(options)

img = cv2.imread('what.jpg', cv2.IMREAD_COLOR)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
result = tfnet.return_predict(img)
label = result[0]['label']
print(label)
tl = (result[0]['topleft']['x'],result[0]['topleft']['y'])
br = (result[0]['bottomright']['x'], result[0]['bottomright']['y'])
img = cv2.rectangle(img, tl , br, (0,255,0), 3)
img = cv2.putText(img, label, tl, cv2.FONT_HERSHEY_COMPLEX, 1, (0,0,0), 2)
plt.imshow(img)
plt.show()