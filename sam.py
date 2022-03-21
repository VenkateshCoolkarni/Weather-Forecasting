import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.naive_bayes import GaussianNB

df = pd.read_csv('weather.csv')
print(df)

Numerics = LabelEncoder()

inputs = df.drop('Play',axis='columns')
target = df['Play']
print(target)

inputs['Outlook_n']=Numerics.fit_transform(inputs['Outlook'])
inputs['Temp_n']=Numerics.fit_transform(inputs['Temp'])
inputs['Humidity_n']=Numerics.fit_transform(inputs['Humidity'])
inputs['Windy_n']=Numerics.fit_transform(inputs['Windy'])
print(inputs)

inputs_n = inputs.drop(['Outlook','Temp','Humidity','Windy'],axis='columns')
print(inputs_n)

Classifier=GaussianNB()
Classifier.fit(inputs_n,target)


Classifier.score(inputs_n,target)

Classifier.predict([[1,1,0,0]])

