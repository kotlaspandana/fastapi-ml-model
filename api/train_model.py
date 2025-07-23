import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import joblib
data={
    'math':[70,80,50,30,20,30,10,40,80,65],
    'science':[30,50,60,20,50,80,60,20,60,70],
    'english':[40,50,70,30,20,70,90,80,40,30],
    'result': ['fail','pass','pass','fail','fail','fail','fail','fail','pass','pass']
}
df=pd.DataFrame(data)
df['result']=df['result'].map({'pass':1,'fail':0})
#train the model
x=df[['math','science','english']]
y=df['result']
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)
print(x_test)
#call the model
model=LogisticRegression()
model.fit(x_train,y_train)
joblib.dump(model,'model.pkl')