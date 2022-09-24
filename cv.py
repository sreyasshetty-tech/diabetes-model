
import pandas as pd
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
import pickle
df=pd.read_csv('diabetes.csv')
print(df.head())

x=df.iloc[:,:-1]
y=df.iloc[:,-1]

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=60)

algorithm = GaussianNB()

algorithm.fit(x_train,y_train)

#pred =algorithm.predict([[0,137,40,35,168,43.1,2.288,33]])
#print("the model has predict",pred)

pickle.dump(algorithm,open('model.pkl','wb'))

mp = pickle.load(open('model.pkl','rb'))
pred=mp.predict([[5,116,74,0,0,25.6,0.201,30]])
print(pred)