import cv2
import numpy as np

faceDetect=cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
cam=cv2.VideoCapture(1);

rec=cv2.createLBPHFaceRecognizer();
rec.load("recognizer/trainningData.yml")

id=0
font=cv2.cv.InitFont(cv2.cv.CV_FONT_HERSHEY_SIMPLEX,1,.5,0,2,1)
fontColor= (194,244,66)
while(True):
	ret, img=cam.read();
	gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
	faces=faceDetect.detectMultiScale(gray,1.3,5);

	for(x,y,w,h) in faces:
		cv2.rectangle(img, (x,y),(x+w,y+h),(244,229,66) ,2)
		id, conf=rec.predict(gray[y:y+h, x:x+w])
		
		if (id==1):
			id="Kshitij"
		
		
		print "Hi "+id + " !" 


		cv2.cv.PutText(cv2.cv.fromarray(img),str(id),(x,y+h),font,fontColor);
	cv2.imshow("Face", img);
	if (cv2.waitKey(1) == ord('q')):
		break;


		cam.release()
		cv2.destroyAllwindows() 
