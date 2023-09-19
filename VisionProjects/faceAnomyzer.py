import cv2
import os
import mediapipe as mp

# read image
imgPath = os.path.join('.', 'VisionProjects','face.webp')
faceImg = cv2.imread(imgPath)

# detect face
def blurFace(img):
    mpFaceDetection = mp.solutions.face_detection

    H, W, _ = img.shape 

    with mpFaceDetection.FaceDetection(model_selection=0, min_detection_confidence=0.5) as faceDetection:
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        out = faceDetection.process(imgRGB)

        if out.detections is not None:
            for detection in out.detections:
                location_data = detection.location_data
                bbox = location_data.relative_bounding_box
                
                x1, y1, w, h = bbox.xmin, bbox.ymin, bbox.width, bbox.height

                x1 = int(x1 * W)
                y1 = int(y1 * H)
                w = int(w * W)
                h = int(h * H)
                
                # blur image
                img[y1:y1+h,x1:x1+w] = cv2.blur(img[y1:y1+h,x1:x1+w], (40, 40))
    return img
            

# display image
#faceImg = blurFace(faceImg)
#cv2.imshow('face', img)
#cv2.waitKey(0)

# save image
#cv2.imwrite(os.path.join('.', 'VisionProjects','blurred.webp'), img)

# blurred face video
cam = cv2.VideoCapture(0)

while True:
    ret, frame = cam.read()
    
    frame = blurFace(frame)
    
    cv2.imshow('frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    
cam.release()
cv2.destroyAllWindows()