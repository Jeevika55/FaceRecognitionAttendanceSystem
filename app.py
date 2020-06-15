from flask import Flask, render_template, request, redirect, url_for, session, send_file
import MySQLdb
import os

from flask_mail import Mail, Message
from math import sqrt

from os import listdir
from os.path import isdir, join, isfile, splitext
import pickle
from PIL import Image, ImageFont, ImageDraw, ImageEnhance
import cv2
import time
from PIL import Image
import numpy as np
import pandas as pd
import csv
import datetime


app = Flask(__name__)

APP_ROOT = os.path.dirname(os.path.abspath(__file__))

app.config.update(
    DEBUG = True,
    MAIL_SERVER = 'smtp.gmail.com',
    MAIL_PORT = 465,
    MAIL_USE_SSL = True,
    MAIL_USERNAME = 'abc@gmail.com',
    MAIL_PASSWORD = 'abc123',
    MAIL_DEFAULT_SENDER = 'abc@gmail.com'
    )
mail = Mail(app)

conn = MySQLdb.connect(host="localhost", user="user", password="password", db="db")

@app.route('/')
def index():
    return render_template("index.html")

@app.after_request
def set_response_headers(response):
    response.headers['Cache-Control'] = 'no-cache, no-store, must-revalidate'
    response.headers['Pragma'] = 'no-cache'
    response.headers['Expires'] = '0'
    return response

@app.route('/adminlogin', methods=['POST'])
def adminlogin():
	user = str(request.form["admin"])
	paswd = str(request.form["adminpass"])
	cursor = conn.cursor()
	result = cursor.execute("SELECT * from adminlogin where adminusername=%s and adminpassword=%s", [user, paswd])
	if(result is 1):
		return render_template("task.html")
	else:
		return render_template("index.html", msg="The username or password is incorrect")

@app.route('/lecturerlogin', methods=['POST'])
def lecturerlogin():
	user = str(request.form["lecturer"])
	session['user'] = user
	paswd = str(request.form["lecturerpass"])
	cursor = conn.cursor()
	result = cursor.execute("SELECT * from lecturerlogin where lectureremailid=%s and lecturerpassword=%s",[user,paswd])
	if result:
		name = user.split('@')[0].capitalize()
		return render_template("task1.html", name=name)
	else:
		return render_template("index.html", msg="The username or password is incorrect")

@app.route('/registerlecturer', methods=['POST'])
def register_lecturer():
	return render_template("reglecturer.html", title="Register Lecturer")


@app.route('/listlecturers', methods=['POST'])
def signup():
	email = str(request.form["email"])
	paswd = str(request.form["pass"])
	cursor = conn.cursor()
	cursor.execute("INSERT INTO lecturerlogin (lectureremailid,lecturerpassword) VALUES(%s, %s)",(email,paswd))
	conn.commit()
	cursor = conn.cursor()
	cursor.execute("SELECT * from lecturerlogin")
	data = cursor.fetchall()
	return render_template("listlecturers.html", msg="successfully signup", uname=email, lecturers=data) 

@app.route('/listlect', methods=['POST'])
def listlect():
	cursor = conn.cursor()
	cursor.execute("SELECT * from lecturerlogin")
	data = cursor.fetchall()
	return render_template("listlecturers.html",lecturers=data)

@app.route('/delete/<string:id_data>', methods = ['GET'])
def delete(id_data):
    cur = conn.cursor()
    cur.execute("DELETE FROM lecturerlogin WHERE lectureremailid=%s", [id_data])
    conn.commit()
    return render_template('reglecturer.html')

@app.route('/student', methods=['POST'])
def student():
	return render_template("regstudent.html")

@app.route('/registerstudent', methods=['POST'])
def regstudent():
	rollno = str(request.form["rollno"])
	name = str(request.form["name"])
	email = str(request.form["studentemail"])
	parentemail = str(request.form["parentemail"])
	cursor = conn.cursor()
	result = cursor.execute("SELECT * from student where  studentrollno=%s", [rollno])
	if result == 1:
		return render_template("regstudent.html", sroll=rollno, msg="Student already present")
	cursor.execute("INSERT INTO student (studentrollno, studentname, studentemail, parentemail) VALUES (%s,%s,%s,%s)",(rollno,name,email,parentemail))
	conn.commit()
	return render_template('regstudent.html')

@app.route("/upload", methods=['POST'])
def createfolder_facedatasets_train():
	dataset = os.path.join(APP_ROOT,"datasets\\")
	if not os.path.isdir(dataset):
		os.mkdir(dataset)
	classfolder = str(request.form['class']+"\\")
	target1 = os.path.join(dataset,classfolder)
	if not os.path.isdir(target1):
		os.mkdir(target1)
	session['dataset'] = dataset
	session['classfolder'] = classfolder
	Id = str(request.form["rollno"])
	vid_cam = cv2.VideoCapture(0)
	face_detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
	count = 0
	while(True):
		_, image_frame = vid_cam.read()
		gray = cv2.cvtColor(image_frame, cv2.COLOR_BGR2GRAY)
		faces = face_detector.detectMultiScale(gray, 1.3, 5)
		for (x, y, w, h) in faces:
        		cv2.rectangle(image_frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
        		count += 1
        		cv2.imwrite(os.path.join(target1,str(Id)+'.'+str(count)+".jpg"), gray[y:y+h,x:x+w])
        		cv2.imshow('Frame', image_frame)
		if cv2.waitKey(100) & 0xFF == ord('q'):
       			break
		elif count>50:
        		break
	vid_cam.release()
	cv2.destroyAllWindows()
	cursor = conn.cursor()
	cursor.execute("SELECT studentname from student where studentrollno=%s",[Id])
	name = cursor.fetchone()[0]
	row = [Id , name]
	with open('StudentDetails\StudentDetails.csv','a+') as csvFile:
		writer = csv.writer(csvFile)
		writer.writerow(row)
	csvFile.close()
	recognizer = cv2.face.LBPHFaceRecognizer_create()
	def getImagesAndLabels(path):
		imagePaths = [os.path.join(path,f) for f in os.listdir(path)]
		faceSamples=[]
		Ids = []
		for imagePath in imagePaths:
			PIL_img = Image.open(imagePath).convert('L')
			img_numpy = np.array(PIL_img,'uint8')
			Id = int(str(os.path.split(imagePath)[-1].split(".")[0]))
			print(Id)
			faces = face_detector.detectMultiScale(img_numpy)
			for (x,y,w,h) in faces:
				faceSamples.append(img_numpy[y:y+h,x:x+w])
				Ids.append(Id)
		return faceSamples,Ids
	faces,Ids = getImagesAndLabels(target1)
	recognizer.train(faces, np.array(Ids))
	recognizer.write('trainer/trainer.yml')
	return render_template('regstudent.html')


@app.route('/takeattendance',methods=['POST'])
def takeattendance():
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.read("trainer/trainer.yml")
    harcascadePath = "haarcascade_frontalface_default.xml"
    faceCascade = cv2.CascadeClassifier(harcascadePath);    
    df=pd.read_csv("StudentDetails\StudentDetails.csv")
    cam = cv2.VideoCapture(0)
    font = cv2.FONT_HERSHEY_SIMPLEX        
    col_names =  ['Id','Name','Date','Time','P/A']
    attendance = pd.DataFrame(columns = col_names)    
    while True:
        ret, im =cam.read()
        gray=cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
        faces=faceCascade.detectMultiScale(gray, 1.2,5)    
        for(x,y,w,h) in faces:
            cv2.rectangle(im,(x,y),(x+w,y+h),(225,0,0),2)
            Id, conf = recognizer.predict(gray[y:y+h,x:x+w])                                   
            if(conf < 50):
                ts = time.time()      
                date = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d')
                timeStamp = datetime.datetime.fromtimestamp(ts).strftime('%H:%M:%S')
                aa=df.loc[df['Id'] == Id]['Name'].values
                tt=str(Id)+"-"+aa
                attendance.loc[len(attendance)] = [Id,aa,date,timeStamp,'Present'] 
            else:
                Id='Unknown'                
                tt=str(Id)  
            if(conf > 75):
                noOfFile=len(os.listdir("ImagesUnknown"))+1
                cv2.imwrite("ImagesUnknown\Image"+str(noOfFile) + ".jpg", im[y:y+h,x:x+w])            
            cv2.putText(im,str(tt),(x,y+h), font, 1,(255,255,255),2)        
        attendance=attendance.drop_duplicates(subset=['Id'],keep='first')    
        cv2.imshow('im',im) 
        if (cv2.waitKey(1)==ord('q')):
            break
    ts = time.time()      
    date = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d')
    timeStamp = datetime.datetime.fromtimestamp(ts).strftime('%H:%M:%S')
    Hour,Minute,Second=timeStamp.split(":")
    fileName="Attendance\Attendance_"+date+"_"+Hour+"-"+Minute+".csv"
    attendance.to_csv(fileName,index=False)
    cam.release()
    cv2.destroyAllWindows()
    f1 = pd.read_csv('C:\\Users\\hp\\Desktop\\FinalProject\\StudentDetails\\StudentDetails.csv')
    f2 = pd.read_csv(fileName)
    df = pd.DataFrame(f1[~f1.Id.isin(f2.Id)])
    df['Date'] = date
    df['Time'] = timeStamp
    df['P/A'] = "Absent"
    df.to_csv(fileName, mode='a',index=False, header=False)
    return absentee(fileName)

def absentee(fileName):
	ts = time.time()   
	date = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d')
	timeStamp = datetime.datetime.fromtimestamp(ts).strftime('%H:%M:%S')
	listOfIds = []
	f = open(fileName,'r')
	reader = csv.reader(f, delimiter=',')
	for row in reader:
		if(row[4] == 'Absent'):
			listOfIds.append(row[0])
	cursor = conn.cursor()
	for i in range(len(listOfIds)):
		cursor.execute("SELECT parentemail from student where studentrollno=%s",[listOfIds[i]])
		email = list(cursor.fetchone())
		print(email[0])
		msg = Message('DBIT', recipients= [email[0]])
		msg.html = "Your ward has missed class held on " +date+ " at " +timeStamp+ "."
		mail.send(msg)
	return render_template('task1.html')

@app.route('/viewreport', methods=['POST'])
def viewreport():
	return render_template("viewreport.html")

@app.route('/view',methods=['POST'])
def view():
	excel_date = request.form['date']
	excel_time = request.form['time']
	Hour,Minute=excel_time.split(":")
	excel_dir = APP_ROOT+"\Attendance\Attendance_"+excel_date+"_"+Hour+"-"+Minute+".csv"
	return send_file(excel_dir, mimetype='text/csv',as_attachment=True)

@app.route('/changetask',methods=['POST'])
def changetask():
	return render_template("task.html")

@app.route('/changetask1',methods=['POST'])
def changetask1():
	return render_template("task1.html")

@app.route('/logout',methods=['POST'])
def logout():
	return render_template("index.html",msg1="Logged out please login again")


if(__name__ == '__main__'):
	app.secret_key = 'secretkey'
	app.run(debug=True)