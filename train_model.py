import pandas as pd
import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

# Load and prepare data http://127.0.0.1:5000

exercise = pd.read_csv('exercise.csv')
calories = pd.read_csv('calories.csv')
data = pd.merge(exercise, calories, on='User_ID')
data['Gender'] = data['Gender'].map({'male': 0, 'female': 1})
data.drop(columns=['User_ID'], inplace=True)

X = data.drop(columns=['Calories'])
y = data['Calories']

# Train model
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Save model
joblib.dump(model, 'model/calorie_model.pkl')
print("Model saved!")

