from darkflow.net.build import TFNet
import cv2
#options = {"model": "cfg/yolo.cfg", "load": "bin/yolo.weights", "threshold": 0.1}
options = {"model": "cfg/yolo.cfg", "load": "bin/yolo.weights", "threshold": 0.1}
tfnet = TFNet(options)
imgcv = cv2.imread("./sample_img/sample_dog.jpg")
results = tfnet.return_predict(imgcv)
print(results)

for result in results:
    if(result["confidence"] > 0.5 ):
        cv2.rectangle(imgcv,
                      (result["topleft"]["x"], result["topleft"]["y"]),
                      (result["bottomright"]["x"],
                       result["bottomright"]["y"]),
                      (255, 0, 0), 4)
        text_x, text_y = result["topleft"][
            "x"] - 10, result["topleft"]["y"] - 10

        cv2.putText(imgcv, result["label"], (text_x, text_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2, cv2.LINE_AA)


cv2.namedWindow('image',cv2.WINDOW_NORMAL)
cv2.imshow('image',imgcv)
cv2.waitKey(0)
cv2.destroyAllWindows()