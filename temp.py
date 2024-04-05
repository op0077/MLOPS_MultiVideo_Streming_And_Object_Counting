import cv2
t = cv2.imread("download.jpg")
while(True):
    cv2.imshow('j',t)
    k = cv2.waitKey(0)
    if k & 0xff == ord('q'):
        break
cv2.release()
cv2.destroyAllWindows()