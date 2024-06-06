import cv2

faceClassifier = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"

)
videoCapture = cv2.VideoCapture(0)

def detectBoundingBox(vid):

    grayImage = cv2.cvtColor(vid, cv2.COLOR_BGR2GRAY)

    faces = faceClassifier.detectMultiScale(grayImage, 1.1, 5, minSize=(40,40))

    for(x,y,w,h) in faces:
        cv2.rectangle(vid,(x,y), (x+h,y+h), (100,100,100), 4)
    return faces

while True:
    result, videoFrame = videoCapture.read()
    if result is False:
        break


    faces = detectBoundingBox(videoFrame)

    cv2.imshow("Cam1", videoFrame)

    k = cv2.waitKey(1)
    if k == 27:
        print("ESC pressed,Closing.")
        break
videoCapture.release()
cv2.destroyAllWindows()
