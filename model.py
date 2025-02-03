import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder
import pickle

df = sns.load_dataset('tips')

le = LabelEncoder()
df['Time'] = le.fit_transform(df['time'])

X = df[['total_bill', 'Time', 'size']]  
y = df['tip']  

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)

with open('tip_prediction_model.pkl', 'wb') as model_file:
    pickle.dump(model, model_file)

with open('tip_prediction_model.pkl', 'rb') as model_file:
    loaded_model = pickle.load(model_file)

input_data = pd.DataFrame([[35.50, le.transform(['Dinner'])[0], 3]], columns=['total_bill', 'Time', 'size'])

predicted_tip = loaded_model.predict(input_data)

print(f"Predicted Tip for the input: {predicted_tip[0]:.2f}")
