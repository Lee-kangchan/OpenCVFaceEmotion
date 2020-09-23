import cv2
import numpy as np
import base64
import requests , json
from os import listdir
from os.path import isfile, join
import os
from azure.cognitiveservices.vision.face import FaceClient
from msrest.authentication import CognitiveServicesCredentials
from azure.cognitiveservices.vision.face.models import TrainingStatusType, Person, SnapshotObjectType, OperationStatusType
##### 여기서부터는 Part2.py와 동일

subscription_key = 'c7c50aa885e6474ba4a72a4c52c3bd6f'
assert subscription_key

data_path = 'faces/'
onlyfiles = [f for f in listdir(data_path) if isfile(join(data_path,f))]
Training_Data, Labels = [], []

face_api_url = 'https://devchan.cognitiveservices.azure.com/face/v1.0/detect'
headers = {'Ocp-Apim-Subscription-Key' : subscription_key}
params = {
    'returnFaceId': 'true',
    'returnFaceLandmarks': 'false',
    'returnFaceAttributes': 'age,gender,headPose,smile,facialHair,glasses,emotion,hair,makeup,occlusion,accessories,blur,exposure,noise',
}


for i, files in enumerate(onlyfiles):
    image_path = data_path + onlyfiles[i]
    images = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if images is None:
        continue
    Training_Data.append(np.asarray(images, dtype=np.uint8))
    Labels.append(i)
if len(Labels) == 0:
    print("There is no data to train.")
    exit()
Labels = np.asarray(Labels, dtype=np.int32)
model = cv2.face.LBPHFaceRecognizer_create()
model.train(np.asarray(Training_Data), np.asarray(Labels))
print("Model Training Complete!!!!!")
#### 여기까지 Part2.py와 동일

#### 여긴 Part1.py와 거의 동일
face_classifier = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

def face_detector(img, size = 0.5):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_classifier.detectMultiScale(gray,1.3,5)
    if faces is():
        return img,[]
    for(x,y,w,h) in faces:
        cv2.rectangle(img, (x,y),(x+w,y+h),(0,255,255),2)
        roi = img[y:y+h, x:x+w]
        roi = cv2.resize(roi, (200,200))
    return img,roi   #검출된 좌표에 사각 박스 그리고(img), 검출된 부위를 잘라(roi) 전달
#### 여기까지 Part1.py와 거의 동일
#카메라 열기
cap = cv2.VideoCapture(0)

count =0
while True:
    #카메라로 부터 사진 한장 읽기
    print(count)
    ret, frame = cap.read()
    # 얼굴 검출 시도
    image, face = face_detector(frame)
    try:
        #검출된 사진을 흑백으로 변환
        face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)

        #위에서 학습한 모델로 예측시도
        result = model.predict(face)
        #result[1]은 신뢰도이고 0에 가까울수록 자신과 같다는 뜻이다.
        if result[1] < 500:
            #????? 어쨋든 0~100표시하려고 한듯
            confidence = int(100*(1-(result[1])/300))
            # 유사도 화면에 표시
            display_string = str(confidence)+'% Confidence it is user'
        cv2.putText(image,display_string,(100,120), cv2.FONT_HERSHEY_COMPLEX,1,(250,120,255),2)
        #75 보다 크면 동일 인물로 간주해 UnLocked!
        if confidence < 150:
            count= count+1
            if count > 30:
                url = 'C:/Users/abc/AllWorkBench/FaceRecognitionWorkbench/opencv/data/test.jpg'
                cv2.imwrite(url, face)
                base64e = base64.b64encode(open(url, 'rb').read())
                data = requests.post("https://api.imgbb.com/1/upload?expiration=600&key=19384f89e631caf75ee33afbf564174b", data={"image" : base64e})

                print(data.text)
                data2 = json.dumps(data.text,sort_keys=True, indent=2, separators=(',', ': '));
                print(data2)
                data3 = json.loads(data2);
                print(data3)
                url = "https://i.ibb.co/yFrtC1x/73a8e3535b25.jpg"

                print(url)
                response = requests.post(face_api_url, params=params,
                                         headers=headers, json={"url": url})

                print(json.dumps(response.json()))
                count = 0;

    except :
        #얼굴 검출 안됨
        cv2.putText(image, "Face Not Found", (250, 450), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 0, 0), 2)
        cv2.imshow('Face Cropper', image)
        pass
    if cv2.waitKey(1)==13:
        break
cap.release()
cv2.destroyAllWindows()