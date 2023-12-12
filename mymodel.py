import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")

df=pd.read_csv("Heart.csv")

df.drop(['Unnamed: 0'],axis=1,inplace=True)
df.dropna(how="all",subset=['Ca'],inplace=True)
df.dropna(how="all",subset=['Thal'],inplace=True)

numcol=[] 
catcol=[]
for i in df.dtypes.index:
  if df.dtypes[i]=="object":
    catcol.append(i)
  else:
    numcol.append(i)
numcol
catcol

features=df[['Age','Sex','RestBP','Chol','Fbs','RestECG','MaxHR','ExAng','Oldpeak','Slope','Ca']]
from scipy.stats import zscore
z=abs(zscore(features))
newdf =df[(z<=3).all(axis =1)]

from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
newdf['AHD']=le.fit_transform(newdf['AHD'])

skew1 = ['Age','Sex','RestBP','Chol','Fbs','RestECG','MaxHR','ExAng','Oldpeak','Slope','Ca']
from sklearn.preprocessing import PowerTransformer
pt=PowerTransformer(method ="yeo-johnson")
newdf[skew1]=pt.fit_transform(newdf[skew1].values)


from sklearn.preprocessing import OrdinalEncoder 
oe =OrdinalEncoder()
newdf[catcol] = oe.fit_transform(newdf[catcol])

x=newdf.drop("AHD", axis =1)
y=newdf["AHD"]


from sklearn.preprocessing import StandardScaler
sc =StandardScaler()
x= sc.fit_transform(x)


from sklearn.model_selection import train_test_split
xtrain,xtest,ytrain,ytest = train_test_split(x,y,test_size=0.4,random_state =1)
from sklearn.naive_bayes import MultinomialNB,GaussianNB,BernoulliNB
from sklearn.metrics import classification_report,confusion_matrix,accuracy_score

def mymodel(model):
    model.fit(xtrain,ytrain)
    ypred=model.predict(xtest)
    
    train=model.score(xtrain,ytrain)
    test=model.score(xtest,ytest)
    
    #print(f"Traning accuracy:{train}\n Testing accuracy:{test}\n\n")
    #print(confusion_matrix(ytest,ypred))
    #print(classification_report(ytest,ypred))
    #print(accuracy_score(ytest,ypred))
    return model
bnb=mymodel(BernoulliNB(alpha=0.01,binarize=0.0,fit_prior=True,class_prior=None))

def mymodel(model):
    model.fit(xtrain,ytrain)
    return model
def makepredict():
    bnb=BernoulliNB()
    model=mymodel(bnb)
    return model