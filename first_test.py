import cv2

cap=cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
while(cap.isOpened()):
    #成功则ret为True, frame是数组
    ret,frame=cap.read()
    try:
        image_np = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    except:
        print("Error converting to RGB")
    print(frame)
    cv2.imshow("Video",frame)

    if cv2.waitKey(10)==ord("q"):
        break

cap.release()
cv2.destroyAllWindows()