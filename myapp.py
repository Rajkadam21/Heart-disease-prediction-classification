from flask import Flask,render_template,request
from mymodel import *
f1=Flask(__name__)

@f1.route("/")
def home():
    return render_template("page1.html")

@f1.route("/getpredict",methods=['GET','POST'])

def getpredict():
    if request.method=='POST':
        Age=request.form['Age']
        Sex=request.form['Sex']
        Chestpain=request.form['Chestpain']
        RestBP=request.form['RestBP']
        Chol=request.form['Chol']
        Fbs=request.form['Fbs']
        RestECG=request.form['RestECG']
        MaxHR=request.form['MaxHR']
        ExAng=request.form['ExAng']
        Oldpeak=request.form['Oldpeak']
        Slope=request.form['Slope']
        Ca=request.form['Ca']
        Thal=request.form['Thal']
         
        print(Age)
        print(Sex)
        print(Chestpain)
        print(RestBP)
        print(Chol)
        print(Fbs)
        print(RestECG)
        print(MaxHR)
        print(ExAng)
        print(Oldpeak)
        print(Slope)
        print(Ca)
        print(Thal)

        newobs=np.array([[Age,Sex,Chestpain,RestBP,Chol,Fbs,RestECG,MaxHR,ExAng,Oldpeak,Slope,Ca,Thal]],dtype=float)
        model=makepredict()
        yp=model.predict(newobs)[0]
        result = "Patient has heart Disease" if yp == 1 else 'Patient is totally fine'
        return render_template("page2.html",data=result)
    
if __name__=="__main__":
    f1.run(debug=True)