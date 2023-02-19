import cv2

# Load the cascade
face_cascade = cv2.CascadeClassifier('C:/Users/Thilaga/Documents/ML_SupervisedLearning/Face_Detection/haarcascade_frontalface_default.xml')
# Read the input image
img = cv2.imread('C:/Users/Thilaga/Documents/ML_SupervisedLearning/Face_Detection/img.jpeg')
# Convert into grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# Detect faces
faces = face_cascade.detectMultiScale(gray, 1.1, 4)
print(faces)
# Draw rectangle around the faces
for (x, y, w, h) in faces:
    cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 10)
# Display the output
cv2.imshow('img', img)
cv2.waitKey()