import cv2
import numpy as np
import os

path = 'ImageQuery'
orb = cv2.ORB_create(nfeatures=1000)
images = []
classNames = []
myList = os.listdir(path)
print('Total classes detected:', len(myList))
for cl in myList:
    imgCur = cv2.imread(f'{path}/{cl}', 0)
    images.append(imgCur)
    classNames.append(os.path.splitext(cl)[0])
print(classNames)

def finDes(images):
    desList = []
    for img in images:
        kp, des = orb.detectAndCompute(img, None)
        desList.append(des)
    return desList


def findId(img, desList, thres=15):
    kp2, des2 = orb.detectAndCompute(img, None)
    bf = cv2.BFMatcher()
    matchList = []
    finalValue = -1
    try:
        for des in desList:
            matches = bf.knnMatch(des, des2, k=2)
            good = []
            for m, n in matches:
                if m.distance < 0.75 * n.distance:
                    good.append([m])
            matchList.append(len(good))
    except:
        pass

    # print(matchList)

    if len(matchList) != 0:
        if max(matchList) > thres:
            finalValue = matchList.index(max(matchList))
        return finalValue


desList = finDes(images)
print(len(desList))

cap = cv2.VideoCapture(0)

while True:

    success, img2 = cap.read()
    imgOriginal = img2.copy()
    img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    id = findId(img2, desList)
    if id!= -1:
        cv2.putText(imgOriginal, classNames[id], (50,50), cv2.FONT_HERSHEY_COMPLEX, 1, (0,0,255), 2)

    cv2.imshow('img2', imgOriginal)
    cv2.waitKey(1)


# img1 = cv2.imread('ImageQuery/imgquery200.jpg', 0)
# img2 = cv2.imread('ImageTrain/curr_img.jpg', 0)

# kp1, des1 = orb.detectAndCompute(img1, None)
# kp2, des2 = orb.detectAndCompute(img2, None)

# imgKp1 = cv2.drawKeypoints(img1, kp1, None)
# imgKp2 = cv2.drawKeypoints(img2, kp2, None)

# bf = cv2.BFMatcher()
# matches = bf.knnMatch(des1, des2, k=2)
# good = []
# for m,n in matches:
    # if m.distance < 0.75 * n.distance:
        # good.append([m])
# print(len(good))
# img3 = cv2.drawMatchesKnn(img1, kp1, img2, kp2, good, None, flags=2)

# cv2.imshow('kp1', imgKp1)
# cv2.imshow('kp2', imgKp2)
# cv2.imshow('img1', img1)
# cv2.imshow('img2', img2)
# cv2.imshow('img3', img3)
# cv2.waitKey(0)
