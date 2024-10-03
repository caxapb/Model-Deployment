import pandas as pd 
from sklearn.model_selection import train_test_split
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import joblib
import os

project_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
file_path = project_dir + '/data/mushroom_cleaned.csv'

df = pd.read_csv(file_path)
df.drop_duplicates(inplace=True)

x = df.drop('class' , axis=1)
y = df['class']

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(x_train)
X_test_scaled = scaler.transform(x_test)

model = RandomForestClassifier()
model.fit(x_train, y_train)

model_path = project_dir + '/models/random_forest_model.pkl'

joblib.dump(model, model_path)
