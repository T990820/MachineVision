import cv2 as cv
face_cascade = cv.CascadeClassifier(cv.data.haarcascades+'haarcascade_frontalface_default.xml')
eye_cascade = cv.CascadeClassifier(cv.data.haarcascades+'haarcascade_eye.xml')
smile_cascade = cv.CascadeClassifier(cv.data.haarcascades+'haarcascade_smile.xml')
cap = cv.VideoCapture(0)  # 参数是0，表示打开笔记本的内置摄像头，参数是视频文件路径则打开视频，如cap = cv2.VideoCapture(“../test.avi”)
cap.set(3, 1080)          # 设置显示界面宽度
cap.set(4, 640)           # 设置显示界面高度
cap.set(10, 100)
while True:
    success, img = cap.read()
    faces = face_cascade.detectMultiScale(img, scaleFactor=1.3, minNeighbors=5)
    for (x, y, w, h) in faces:
        cv.putText(img, "face", (x, y), cv.FONT_ITALIC, 1, (255, 0, 0), 2)
        img = cv.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)
        face_area = img[y:y+h, x:x+w]
        smiles = smile_cascade.detectMultiScale(face_area, scaleFactor=1.3, minNeighbors=70)
        for (sx, sy, sw, sh) in smiles:
            img = cv.rectangle(img, (sx+x, sy+y), (sx+x+sw, sy+y+sh), (0, 0, 255), 2)
            cv.putText(img, "smile", (sx+x, sy+y), cv.FONT_ITALIC, 1, (0, 0, 255), 2)
        eyes = eye_cascade.detectMultiScale(img, scaleFactor=1.3, minNeighbors=8)
        for (ex, ey, ew, eh) in eyes:
            img = cv.rectangle(img, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2)
            cv.putText(img, "eye", (ex, ey), cv.FONT_ITALIC, 1, (0, 255, 0), 2)
    cv.imshow('output', img)
    if cv.waitKey(1) & 0xff == ord('q'):  # 等待用户1ms，如果用户按下了q则执行break
        break
cap.release()

