from flask import Flask, render_template, request, redirect, url_for
from werkzeug.utils import secure_filename
import pickle
import json

import pandas as pd
import numpy as np
from sklearn import tree

from flask import send_from_directory, Response

from keras.models import load_model
from keras import backend as K

from waitress import serve



app = Flask(__name__)

@app.route('/')
def upload_file1():
    return render_template('home.html')

#--DOWNLOADZWEI--DOWNLOADZWEI--DOWNLOADZWEI--DOWNLOADZWEI--DOWNLOADZWEI--DOWNLOADZWEI--DOWNLOADZWEI---------------------------------------------------------------------------------
@app.route('/uploaderzwei', methods = ['GET', 'POST'])
def upload_filezwei():
    if request.method == 'POST':
       df = pd.read_csv(request.files.get('file'), sep=',',header=0, encoding='TIS-620')
       hpno = df['Account']
       df = df.drop('Account', axis=1) #ลบคอลั่มAccount

       actionCode = request.form.get('actionCode')  #รับรหัสว่าใช้โมเดลอะไรในการทำนาย
       
       #------------------getDUMMY สำหรับเทียบให้เหมือนกับโมเดล Exโมเดลมี 56แอตทริบิวต์หลังจากทำone hot จึงต้องทำให้ไฟล์ใหม่แปลงเป็นone hotให้ได้สเกลของแอตทริบิวต์เท่ากับของโมเดลนั่นคือ56แอตทริบิวต์ เป็นต้น----------------------------#
       ord1 = pd.read_csv('ชื่อไฟล์.csv',sep=',',header=0, encoding='TIS-620')
       for column in ord1.select_dtypes(include=[np.object]).columns:
           df[column] = df[column].astype('category', categories = ord1[column].unique())
       #---------------------------------------------------------------------------------#
       oh = pd.get_dummies(df) #ทำข้อมูลแปลงให้เป็นแบบ ONE HOT

       #-----------IF----------------#
       if actionCode == "1":
            #------------Neural Network-----------------#
            md = load_model('ชื่อโมเดล')  #เรียกใช้โมเดลNeural Netwok
            y_pred = md.predict_classes(oh)
            y_proba = md.predict_proba(oh)
            K.clear_session()

            aciYES = []      #------เก็บคำตอบYES ปล.ไม่ได้ใช้แล้ว
            AccountYes = []  #------เก็บคำตอบรหัสACCOUNT
            proYes = []      #------เก็บคำตอบความน่าจะเป็น
            #--คัดเฉพาะคำตอบ YES--#
            for x in range(len(y_pred)):
                if y_pred[x][0] == 1:
                    aciYES.append(hpno[x])
                    AccountYes.append(hpno[x])
                    proYes.append(y_proba[x][0])
            return render_template('result.html', valuxs=aciYES, Acc=AccountYes, proB=proYes, mdname='Neural Network')

       #---เข้า Decision Tree---#
       elif actionCode == "2":      #---เข้า Decision Tree---#
           md = pickle.load(open('่ชื่อโมเดล', 'rb'))
           y_pred = md.predict(oh)
           y_proba = md.predict_proba(oh)

           #---เลือกprobaที่มากที่สุดไว้แสดง---#
           y_proba_max = []
           for x in range(len(y_proba)):
               if y_proba[x][0]>y_proba[x][1]:
                   y_proba_max.append(y_proba[x][0])
               else: y_proba_max.append(y_proba[x][1])

           aciYES = []      #------เก็บคำตอบYES ปล.ไม่ได้ใช้แล้ว
           AccountYes = []  #------เก็บคำตอบรหัสACCOUNT
           proYes = []      #------เก็บคำตอบความน่าจะเป็น
           #--คัดเฉพาะคำตอบ YES--#
           for x in range(len(y_pred)):
               if y_pred[x] == "YES":
                   aciYES.append(hpno[x])
                   AccountYes.append(hpno[x])
                   proYes.append(y_proba_max[x])
           return render_template('result.html', valuxs=aciYES, Acc=AccountYes, proB=proYes, mdname='Decision Tree')

       #---เข้า Naive Bayes---#
       elif actionCode == "3":  #---เข้า Naive Bayes---#
           md = pickle.load(open('ชื่อโมเดล', 'rb'))
           y_pred = md.predict(oh)
           y_proba = md.predict_proba(oh)

           #---เลือกprobaที่มากที่สุดไว้แสดง---#
           y_proba_max = []
           for x in range(len(y_proba)):
               if y_proba[x][0]>y_proba[x][1]:
                   y_proba_max.append(y_proba[x][0])
               else: y_proba_max.append(y_proba[x][1])
           
           aciYES = []      #------เก็บคำตอบYES ปล.ไม่ได้ใช้แล้ว
           AccountYes = []  #------เก็บคำตอบรหัสACCOUNT
           proYes = []      #------เก็บคำตอบความน่าจะเป็น
           #--คัดเฉพาะคำตอบ YES--#
           for x in range(len(y_pred)):
               if y_pred[x] == "YES":
                   aciYES.append(hpno[x])
                   AccountYes.append(hpno[x])
                   proYes.append(y_proba_max[x])
           return render_template('result.html', valuxs=aciYES, Acc=AccountYes, proB=proYes, mdname='Naive Bayes')
       #---8yho----#

       #---เข้า Compair3Model---#
       elif actionCode == "4":  #---เข้า Compair3Model---#
           account = [] #เก็บข้อมูล ACCOUNT
           NN_ans = []#เก็บคำตอบของNeural Network
           DT_ans = []#เก็บคำตอบของDecision Tree
           NB_ans = []#เก็บคำตอบของNaive Bayes
           FN_ans = [] #คำตอบสุดท้ายว่าสำเร็จ

           #-----------------เรียก3โมเดล----------------------#
           #-----Neural Network------#
           md = load_model('ชื่อโมเดล')
           NN_pred = md.predict_classes(oh)
           NN_proba = md.predict_proba(oh)
           NN_predTrans = []
           for s in range(len(NN_pred)):
               if NN_pred[s][0] == 1:
                   NN_predTrans.append("YES")
               else: NN_predTrans.append("NO")
           K.clear_session()
           #-----Neural Network-------#

           #-----Decision Tree-------#
           loaded_model_DT = pickle.load(open('ชื่อโมเดล', 'rb'))
           DT_pred = loaded_model_DT.predict(oh)
           DT_proba = loaded_model_DT.predict_proba(oh)
           #-----Decision Tree-------#

           #-----Naive Bayes-------#
           loaded_model_NB = pickle.load(open('ชื่อโมเดล', 'rb'))
           NB_pred = loaded_model_NB.predict(oh)
           NB_proba = loaded_model_NB.predict_proba(oh)
           #-----Naive Bayes-------#
           #-----------------END เรียก3โมเดล----------------------#

           #------ลูป เปรียบเทียบคำตอบ---------------------------#
           for x in range(len(NN_pred)):
               tempN = 0    #เก็บค่าคำตอบของแต่ละโมเดล
               #ตรวจสอบคำตอบของNeural Network
               if(NN_predTrans[x] == "YES"):
                   tempN += 1
               elif(NN_predTrans[x] == "NO"):
                   tempN -= 1

               #ตรวจสอบคำตอบของDecision Tree
               if(DT_pred[x] == "YES"):
                   tempN += 1
               elif(DT_pred[x] == "NO"):
                   tempN -= 1

               #ตรวจสอบคำตอบของNaive Bayes
               if(NB_pred[x] == "YES"):
                   tempN += 1
               elif(NB_pred[x] == "NO"):
                   tempN -= 1


               if(tempN > 0):
                   account.append(hpno[x])
                   NN_ans.append(NN_predTrans[x])
                   DT_ans.append(DT_pred[x])
                   NB_ans.append(NB_pred[x])
                   rate = (NN_proba[x][0]+DT_proba[x][1]+NB_proba[x][1])/3
                   FN_ans.append(rate*100)
           return render_template('resultForCompair3Model.html',account=account, nn=NN_ans, dt=DT_ans, nb=NB_ans, fn=FN_ans, mdname='Compare 3 Model', massage='รายชื่อลูกค้าที่ระบบทำนายว่าจะเสนอขายสำเร็จ')

#--DOWNLOADZWEI---DOWNLOADZWEI---DOWNLOADZWEI---DOWNLOADZWEI---DOWNLOADZWEI----DOWNLOADZWEI------DOWNLOADZWEI------------------------------------------------------------------#

#---------------MANUAL---------------------- MANUAL INPUT --------------------------MANUAL---------------------#
@app.route('/upload_manual', methods = ['GET', 'POST'] )
def manual_input():
    if request.method == 'POST':
        dfx = [[request.form.get('REGION'), request.form.get('AGE'), request.form.get('YEAR_OF_PRODUCT'), request.form.get('TYPE_PRODUCT'), request.form.get('NEW_USED_'), request.form.get('COM_ROUND'), request.form.get('T25_COM_TYPE_COVERAGE'), request.form.get('COM_CONFIRM'), request.form.get('T25_COM_INS_CODE'), request.form.get('CLAIM_CON'), request.form.get('INS_PAY_TYPE'), request.form.get('INS_PAY_BY')]]
        df = pd.DataFrame(dfx, columns=['REGION','AGE','YEAR_OF_PRODUCT','TYPE_PRODUCT','NEW_USED_','COM_ROUND','T25_COM_TYPE_COVERAGE','COM_CONFIRM','T25_COM_INS_CODE','CLAIM_CON','INS_PAY_TYPE','INS_PAY_BY'])
        hpno = request.form.get('hpno')
        #------------------getDUMMY สำหรับเทียบให้เหมือนกับโมเดล----------------------------#
        ord1 = pd.read_csv('ชื่อไฟล์.csv',sep=',',header=0, encoding='TIS-620')
        for column in ord1.select_dtypes(include=[np.object]).columns:
            df[column] = df[column].astype('category', categories = ord1[column].unique())
        #---------------------------------------------------------------------------------#
        oh = pd.get_dummies(df)
        
        account = [] #เก็บข้อมูล ACCOUNT
        NN_ans = []#เก็บคำตอบของNeural Network
        DT_ans = []#เก็บคำตอบของDecision Tree
        NB_ans = []#เก็บคำตอบของNaive Bayes
        FN_ans = [] #คำตอบสุดท้ายว่าสำเร็จ
        massage = '' #ข้อความหัวข้อ

        #-----------------เรียก3โมเดล----------------------#
        #-----Neural Network------#
        md = load_model('ชื่อโมเดล')
        NN_pred = md.predict_classes(oh)
        NN_proba = md.predict_proba(oh)
        NN_predTrans = []
        for s in range(len(NN_pred)):
            if NN_pred[s][0] == 1:
                NN_predTrans.append("YES")
            else: NN_predTrans.append("NO")
        K.clear_session()
        

        #-----Decision Tree-------#
        loaded_model_DT = pickle.load(open('ชื่อโมเดล', 'rb'))
        DT_pred = loaded_model_DT.predict(oh)
        DT_proba = loaded_model_DT.predict_proba(oh)
        

        #-----Naive Bayes-------#
        loaded_model_NB = pickle.load(open('ชื่อโมเดล', 'rb'))
        NB_pred = loaded_model_NB.predict(oh)
        NB_proba = loaded_model_NB.predict_proba(oh)
        
        #-----------------END เรียก3โมเดล----------------------#

        #------ลูป เปรียบเทียบคำตอบ---------------------------#
        for x in range(len(NN_pred)):
            tempN = 0
            if(NN_predTrans[x] == "YES"):
                tempN += 1
            elif(NN_predTrans[x] == "NO"):
                tempN -= 1

            if(DT_pred[x] == "YES"):
                tempN += 1
            elif(DT_pred[x] == "NO"):
                tempN -= 1

            if(NB_pred[x] == "YES"):
                tempN += 1
            elif(NB_pred[x] == "NO"):
                tempN -= 1

            if(tempN > 0):
                account.append(hpno)
                NN_ans.append(NN_predTrans[x])
                DT_ans.append(DT_pred[x])
                NB_ans.append(NB_pred[x])
                rate = (NN_proba[x][0]+DT_proba[x][1]+NB_proba[x][1])/3
                FN_ans.append(rate*100)
                massage = 'ระบบทำนายว่า สำเร็จ'
            elif(tempN < 0):
                account.append(hpno)
                NN_ans.append(NN_predTrans[x])
                DT_ans.append(DT_pred[x])
                NB_ans.append(NB_pred[x])
                rate = (NN_proba[x][0]+DT_proba[x][0]+NB_proba[x][0])/3
                FN_ans.append(rate*100)
                massage = 'ระบบทำนายว่า ไม่สำเร็จ'
        return render_template('resultForCompair3Model.html',account=account, nn=NN_ans, dt=DT_ans, nb=NB_ans, fn=FN_ans, mdname='Compare 3 Model', massage=massage)
#---MANUAL----MANUAL-----MANUAL---MANUAL-----MANUAL----MANUAL----MANUAL----MANUAL------------------------------------------------------------------#
@app.route("/App-HealthCheck")
def healthcheck():
    return ('Check')

@app.route("/Warning")
def neededData():
    return render_template('warning.html')

@app.route("/Log")
def showlog():
    return render_template('Log.html')

@app.route("/Preprocess")
def showPre():
    return render_template('Preprocess.html')

if __name__ == '__main__':
   app.run(port='8080', host='0.0.0.0')
