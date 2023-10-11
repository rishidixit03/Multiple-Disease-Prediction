#import required packages
from flask import Flask, render_template, request
from flask.helpers import flash

#import jsonify
import requests
import pickle
import numpy as np
import sklearn
from sklearn.preprocessing import StandardScaler

#create a Flask object
app = Flask("health-app")


#load the ml model which we have saved earlier in .pkl format

class OutOfBounds(Exception):
    pass

#define the route(basically url) to which we need to send http request
#HTTP GET request method
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/home_page')
def home_page():
    return render_template('index.html')

@app.route('/diabetes_details')
def diabetes_details():
    return render_template('diabetes_details.html')

@app.route('/heart_details')
def heart_details():
    return render_template('heart_details.html')

@app.route('/cancer_details')
def cancer_details():
    return render_template('cancer_details.html')

# @app.route('/kidney_details')
# def kidney_details():
#     return render_template('kidney_details.html')
    
@app.route('/predict_cancer', methods=['GET','POST'])
def predict_cancer():
    
    if request.method == 'POST':
     model_cancer = pickle.load(open('breast_cancer.pkl', 'rb'))
     try:
        
        tex = float(request.form['texture_mean'])
        par = float(request.form['perimeter_mean'])
        smooth = float(request.form['smoothness_mean'])
        compact = float(request.form['compactness_mean'])
        sym = float(request.form['symmetry_mean'])


        if tex<9.71 or tex>39.3 or par<43.8 or par>189 or smooth<0.05 or smooth>0.61 or compact<0.02 or compact>0.35 or sym<0.11 or sym>0.3:
            raise OutOfBounds
             
        
        
        prediction = model_cancer.predict([[tex,par,smooth,compact,sym]])
        if prediction == 1:
            return render_template('cancer_output.html',prediction_text="Sorry, you are cancerous!", advice ="Do visit a Doctor", Pre="Precautons -", p1="1. Limit alcohol."
             ,p2="2. Maintain a healthy weight.", p3="3. Be physically active." , p4="4. Breast-feed.", p5="5. Limit postmenopausal hormone therapy. ")
        
        #condition for prediction when values are valid
        if prediction==0:
            return render_template('unappropriate.html',prediction_text="You are non-cancerous")
        
     except ValueError:
            return render_template('unappropriate.html',prediction_text="Please fill the appropriate values according to given data type!")


     except OutOfBounds:
            return render_template('unappropriate.html',prediction_text="Please fill out the values in the given range")        

       
    else:
        return render_template('cancer.html')



@app.route('/predict_diabetes', methods=['GET','POST'])
def predict_diabetes():
    if request.method == 'POST':
     model_diabetes = pickle.load(open('diabetes.pkl', 'rb'))

     try:
        
        Pregnancies = int(request.form['Pregnancies'])
        Glucose = int(request.form['Glucose'])
        BloodPressure = int(request.form['BloodPressure'])
        SkinThickness = int(request.form['SkinThickness'])
        Insulin = int(request.form['Insulin'])
        BMI = float(request.form['BMI'])
        DiabetesPedigreeFunction = float(request.form['DiabetesPedigreeFunction'])
        Age = int(request.form['Age'])
        
        
        if Pregnancies<0 or Glucose<0 or Glucose>199 or BloodPressure<0 or BloodPressure>122 or SkinThickness<0 or SkinThickness>99 or Insulin<0 or Insulin>846 or BMI<0 or DiabetesPedigreeFunction<0.08 or DiabetesPedigreeFunction>2.42 or Age<0:
            raise OutOfBounds

        
        prediction = model_diabetes.predict([[Pregnancies, Glucose,BloodPressure,SkinThickness,Insulin,BMI,DiabetesPedigreeFunction,Age]])
        
        if prediction == 1:
            return render_template('diabetes_output.html',prediction_text="Sorry, you have diabetes!", BMI = BMI, BloodPressure = BloodPressure ,  advice ="Do visit a Doctor", Pre="Precautons -", p1="1. Keep your blood pressure and cholesterol under control"
             ,p2="2. Maintain a healthy weight.", p3="3. Be physically active." , p4="4. Eat healthy fats", p5="5. Reduce your alcohol consumption")
        
        #condition for prediction when values are valid
        if prediction==0:
            return render_template('unappropriate.html',prediction_text="You don't have diabetes" )
        
     except ValueError:
            return render_template('unappropriate.html',prediction_text="Please fill the appropriate values according to given data type!")

     except OutOfBounds:
            return render_template('unappropriate.html',prediction_text="Please fill out the values in the given range")            

       
    else:
        return render_template('diabetes.html')



        

@app.route('/predict_heart', methods=['GET','POST'])
def predict_heart():
    if request.method == 'POST':
     model_heart = pickle.load(open('heart.pkl', 'rb'))

     try:
      

        age= float(request.form['age'])
        sex = int(request.form['sex'])
        cp = float(request.form['cp'])
        trestbps = float(request.form['trestbps'])
        chol  = float(request.form['chol'])
        fbs = int(request.form['fbs'])
        restecg = float(request.form['restecg'])
        thalach = float(request.form['thalach'])
        exang = int(request.form['exang'])
        oldpeak = float(request.form['oldpeak'])
        slope = float(request.form['slope'])
        ca  = float(request.form['ca'])
        thal  =float(request.form['thal'])
        
        
        if age<0 or sex<0 or sex>1 or cp<0 or cp>3 or chol<126 or chol>564 or fbs<0 or fbs>1 or restecg<0 or restecg>2 or thalach<71 or thalach>202 or exang<0 or exang>1 or oldpeak<0 or oldpeak>6.2 or slope<0 or slope>2 or ca<0 or ca>3 or thal<0 or thal>3:
            raise OutOfBounds 
        
        
        prediction = model_heart.predict([[age,sex,cp,trestbps,chol,fbs,restecg,thalach,exang,oldpeak,slope,ca,thal]])
        if prediction == 1:
            return render_template('heart_output.html',prediction_text="Sorry, you have heart disease!", BP = trestbps, Chol = chol, sex = sex, age = age, HB = thalach,  advice ="Do visit a Doctor", Pre="Precautons -", p1="1. Eat a healthy, balanced diet"
             ,p2="2. Maintain a healthy weight.", p3="3. Be physically active." , p4="4. Reduce your alcohol consumption", p5="5. Give up smoking")
        
        #condition for prediction when values are valid
        if prediction==0:
            return render_template('unappropriate.html',prediction_text="You don't have heart-disease!")
        
     except ValueError:
            return render_template('unappropriate.html',prediction_text="Please fill the appropriate values according to given data type!")

     except OutOfBounds:
            return render_template('unappropriate.html',prediction_text="Please fill out the values in the given range")  

    else:
         return render_template('heart.html') 


# @app.route('/predict_kidney', methods=['GET','POST'])
# def predict_kidney():
#     if request.method == 'POST':
#      model_diabetes = pickle.load(open('kidney.pkl', 'rb'))

#      try:
        
      
#         age = float(request.form['age'])
#         bp = float(request.form['bp'])
#         al = float(request.form['al'])
#         su = float(request.form['su'])
#         rbc = (request.form['rbc'])
#         pc = (request.form['pc'])
#         pcc = (request.form['pcc'])
#         ba = (request.form['ba'])
#         bgr = float(request.form['bgr'])
#         bu = float(request.form['bu'])
#         sc = float(request.form['sc'])
#         pot = float(request.form['pot'])
#         wc = float(request.form['wc'])
#         htn = (request.form['htn'])
#         dm = (request.form['dm'])
#         cad = (request.form['cad'])
#         pe = (request.form['pe'])
#         ane = (request.form['ane'])
        
       
        
#         prediction = model_diabetes.predict([[age, bp, al, su, rbc, pc, pcc, ba, bgr, bu, sc,pot, wc, htn, dm, cad, pe, ane]])
#         if prediction == 1:
#             return render_template('kidney_output.html',prediction_text="Sorry, you have Kidney disease!", albumin = al, WBC = wc, RBG = bgr, SC = sc, BU = bu, K= pot)
        
#         #condition for prediction when values are valid
#         if prediction== 0:
#             return render_template('unappropriate.html',prediction_text="You don't have Kidney disease")
        
#      except ValueError:
#             return render_template('unappropriate.html',prediction_text="Please fill the approriate values!")
                

       
#     else:
#         return render_template('kidney.html')


                
if __name__=="__main__":
    #run method starts our web service
    #Debug : as soon as I save anything in my structure, server should start again
    app.run(debug=True)