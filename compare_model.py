import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler 
from sklearn.neural_network import MLPClassifier 
from sklearn.svm import SVC
from imblearn.over_sampling import SMOTE

# อ่านข้อมูล
data = pd.read_csv('data/marine_data_clean.csv')

# แยกข้อมูลเป็น X และ y
X = data.drop('Criteria', axis=1) 
y = data['Criteria'] 

smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X, y)

# แบ่งข้อมูลเป็นชุดฝึกและชุดทดสอบ
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.3, random_state=42)

# แปลงค่าของ y_test ให้เป็นรหัส (Encoding)
le = LabelEncoder()
y_test_encoded = le.fit_transform(y_test)

# มาตรฐานข้อมูล (Normalization) โดยใช้ StandardScaler
scaler = StandardScaler()  # สร้างตัวปรับมาตรฐาน
X_train_scaled = scaler.fit_transform(X_train)  # ปรับมาตรฐานข้อมูลการฝึก
X_test_scaled = scaler.transform(X_test)  # ปรับมาตรฐานข้อมูลการทดสอบ

# สร้างโมเดลต่างๆ
models = {
    'KNN': KNeighborsClassifier(n_neighbors=5),
    'RandomForest': RandomForestClassifier(n_estimators=100, random_state=42),
    'MLP': MLPClassifier(random_state=42, max_iter=1000),
    'SVM': SVC(random_state=42)
}

# ฝึกและทดสอบแต่ละโมเดล
model_accuracies = {}
for name, model in models.items():
    model.fit(X_train_scaled, y_train)  # ฝึกโมเดล
    y_pred = model.predict(X_test_scaled)  # ทำนายผล
    accuracy = accuracy_score(y_test, y_pred)  # คำนวณความแม่นยำ
    model_accuracies[name] = accuracy  # เก็บผลลัพธ์ความแม่นยำ
    print(f"\n{name} Accuracy: {accuracy:.2f}")
    print(classification_report(y_test, y_pred))

# สร้างกราฟเปรียบเทียบความแม่นยำของแต่ละโมเดล
plt.figure(figsize=(10, 6))
plt.bar(model_accuracies.keys(), model_accuracies.values(), color=['blue', 'green', 'red', 'orange'])
plt.title('Comparison of the accuracy of prediction models')
plt.xlabel('Model')
plt.ylabel('Accuracy')
plt.ylim(0, 1)
plt.show()