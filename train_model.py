import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import joblib

data = pd.read_csv('data/marine_data_clean.csv')

# แยกฟีเจอร์ออกจากป้ายกำกับ (label)
X = data.drop('Criteria', axis=1)  # ฟีเจอร์
y = data['Criteria']  # ป้ายกำกับ

x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

model =RandomForestClassifier(n_estimators=100 ,random_state=42)
model.fit(x_train,y_train)

joblib.dump(model, 'marine_quality_model.pkl')

accuracy = model.score(x_test, y_test)
print(f"Model accuracy: {accuracy:.2f}")