"""
     Titanic problem:
     gender_submission.csv - true output of test.csv
     train.csv - training set
     test.csv - test set   

"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression

class Titanic():
#X-Input Matrix(m*n) y=Output matrix(m*1)
     def __init__(self,X,y):
          self.X=X
          self.y=y
          self.m=X.index.size #No of training examples
          self.X['constant']=np.ones(self.m)
          self.n=X.columns.size #No of features
          self.X_train=X[0:int(self.m*0.7)] #Training examples
          self.y_train=y[0:int(self.m*0.7)]  
          self.X_test=X[int(self.m*0.7):]   #testing examples
          self.y_test=y[int(self.m*0.7):]   
          self.theta=pd.Series(np.zeros((self.n)),index=self.X_train.columns)#Dimension=(n,1)

     def cost(self,X,y):
          m=X.index.size
          h_theta=sigmoid(X.dot(self.theta))#Dimension=(m,1)
          cost=-1/(m)*((y.T).dot(np.log(h_theta))+((np.ones_like(y)-y).T).dot(np.log(np.ones_like(h_theta)-h_theta))) #why cost is a Series 0,some_value
          return cost

     def gradient(self,lambd):
          h_theta=sigmoid(self.X_train.dot(self.theta))#Dimension=(m,1)
          return self.X_train.T.dot(h_theta-self.y_train[0])+lambd*self.theta  
     
     def GradientDescent(self,alpha):
          m=self.X_train.index.size
          cost_list=[]
          for i in range(1000):
               print(self.theta)
               self.theta=self.theta+(alpha/m)*self.gradient(100);
               cost_list.append(self.cost(self.X_train,self.y_train))
          plt.plot(range(1000),cost_list)
          plt.show()

     def accuracy(self):
          y_predicted=sigmoid(self.X_test.dot(self.theta))
          y_predicted=np.where(y_predicted>=0.5,1,0)
          print(y_predicted)
          predicted_correctly=0
          # for i in range(m_test):
          #      if((y_predicted[i]>=0.5 and self.y_test[i]==1) or (y_predicted[i]<0.5 and self.y_test[i]==0)):
          #           predicted_correctly=predicted_correctly+1
          # return predicted_correctly/m_test
          # #print(cost_list)


     def logistic(self):
          logr= LogisticRegression()
          logr.fit(self.X_train,self.y_train)
          y_pred= logr.predict(self.X_test)
          acc_logr= round(logr.score(self.X_train,self.y_train)*100, 2)
          print("Accuracy of Logistic Regression is")
          print(acc_logr)

def sigmoid(x):
     return 1/(1+np.exp(-x))
          

def DataXy(data):
     m=data.index.size
     data['Sex']=np.where(data['Sex']=='male',1,0) #1 for male 0 for female
     mean_female=data[data["Sex"]==0].mean()["Age"]
     mean_male=data[data["Sex"]==1].mean()["Age"]
     mean_age=data["Age"].mean()
     mean_master=data[data["Sex"]==1].mean()["Age"]
     for i in range(m):
          if(np.isnan(data.loc[i,"Age"])):
               listName=data.loc[i,"Name"].split(',')
               if(listName[1].startswith(' Mrs.')):
                    data.set_value(i,"Age",mean_age)
               elif(listName[1].startswith(' Mr.')):
                    data.set_value(i,"Age",mean_age)
               else:
                    data.set_value(i,"Age",0) #Think about this
     y=data['Survived']
     X=data.drop(['PassengerId','Survived','Name','Ticket','Cabin','Embarked'],axis=1)
     return [X,y]

data=pd.read_csv('train.csv')
[X,y]=DataXy(data)
print(X)
print(y)
ship=Titanic(X,y)
#ship.logistic()
ship.GradientDescent(0.0000001)
# x`print(ship.accuracy())     
          