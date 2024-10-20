import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from imblearn.over_sampling import SMOTE

# โหลดข้อมูลจากไฟล์ CSV
df1 = pd.read_csv('data/marine-quality-1_66.csv', encoding='cp874', on_bad_lines='skip')
df2 = pd.read_csv('data/marine-quality-2_66.csv', encoding='cp874', on_bad_lines='skip')

# แสดงคอลัมน์ทั้งหมดจากไฟล์แรก
print("คอลัมน์ทั้งหมดจากไฟล์ marine-quality-1_66.csv:")
print(df1.columns)

# แสดงคอลัมน์ทั้งหมดจากไฟล์ที่สอง
print("\nคอลัมน์ทั้งหมดจากไฟล์ marine-quality-2_66.csv:")
print(df2.columns)

###ลบคอลลัมม์ที่ไม่จำเป็น###
# คอลัมน์ที่ต้องการเก็บ
columns_to_keep = ['Temperature', 'pH', 'EC', 'Salinity', 'DO', 'SS',
                   'Phosphates - phosphorus', 'Total ammonia', 'Nitrate - nitrogen', 
                   'Total coliforms', 'Au', 'Zn', 'Criteria']

# ลบคอลัมน์อื่นๆ และเหลือแค่คอลัมน์ที่ต้องการใน df1
df1_filtered = df1[columns_to_keep]
# ลบคอลัมน์อื่นๆ และเหลือแค่คอลัมน์ที่ต้องการใน df2
df2_filtered = df2[columns_to_keep]
# ตรวจสอบข้อมูลที่เหลืออยู่ใน df1
print("ข้อมูลจากไฟล์ marine-quality-1_66.csv หลังจากลบคอลัมน์:")
print(df1_filtered.head())
# ตรวจสอบข้อมูลที่เหลืออยู่ใน df2
print("\nข้อมูลจากไฟล์ marine-quality-2_66.csv หลังจากลบคอลัมน์:")
print(df2_filtered.head())

###แปลงคลาสเป็นภาษาอังกฤษ###
# สร้าง dictionary สำหรับการแปลงค่า
criteria_mapping = {
    'ดีมาก': 'Very Good',
    'ดี': 'Good',
    'พอใช้': 'Normal',
    'เสื่อมโทรม': 'Bad',
    'เสื่อมโทรมมาก': 'Very Bad'
}
# แปลงค่าคลาสในคอลัมน์ Criteria ของ df1
df1_filtered['Criteria'] = df1_filtered['Criteria'].replace(criteria_mapping)
# แปลงค่าคลาสในคอลัมน์ Criteria ของ df2
df2_filtered['Criteria'] = df2_filtered['Criteria'].replace(criteria_mapping)
# ตรวจสอบข้อมูลที่เหลืออยู่ใน df1
print("ข้อมูลจากไฟล์ marine-quality-1_66.csv หลังจากแปลงค่า Criteria:")
print(df1_filtered.head())
# ตรวจสอบข้อมูลที่เหลืออยู่ใน df2
print("\nข้อมูลจากไฟล์ marine-quality-2_66.csv หลังจากแปลงค่า Criteria:")
print(df2_filtered.head())


### รวม df1 และ df2 เป็น data###
data = pd.concat([df1_filtered, df2_filtered], ignore_index=True)
# แสดงข้อมูลรวม
print("ข้อมูลที่รวมจาก df1 และ df2:")
print(data.head())
# แสดง shape ของ DataFrame data
print("Shape ของ data:", data.shape)
# แสดงข้อมูลของ DataFrame data
print(data.info())

###จัดการค่าที่มี > , < ###
def transform_value(value):
    if isinstance(value, str):
        # ลบเครื่องหมายจุลภาค
        value = value.replace(',', '')
        if value.startswith('>'):
            # ลบเครื่องหมาย > และเพิ่มค่าแบบสุ่ม
            base_value = float(value[1:])
            new_value = round(base_value + np.random.uniform(0.01, 1.0), 2)
            return new_value if new_value >= 0 else 0  # ให้ค่าไม่ติดลบ
        elif value.startswith('<'):
            # ลบเครื่องหมาย < และลดค่าแบบสุ่ม
            base_value = float(value[1:])
            new_value = round(base_value - np.random.uniform(0.01, 1.0), 2)
            return new_value if new_value >= 0 else 0  # ให้ค่าไม่ติดลบ
    return value
# แปลงค่าทุกคอลัมน์ใน DataFrame ที่มีค่าที่เป็น string
for col in data.columns:
    data[col] = data[col].apply(transform_value)
# แสดงผลลัพธ์
print(data.sample(10))
# แสดงข้อมูลของ DataFrame data
print(data.info())

### datatype เป็น Float เว้น Criteria ###
# ฟังก์ชันแปลงคอลัมน์ที่ไม่ใช่ Criteria เป็น float
for col in data.columns:
    if col != 'Criteria':
        data[col] = pd.to_numeric(data[col], errors='coerce')
# แสดงข้อมูลที่แปลงแล้ว
print(data.info())
# แสดงผลลัพธ์
print(data.sample(10))

### ตัดการค่าว่าง ###
print("ค่าที่หายไปในแต่ละคอลัมน์ก่อนการเติมค่า:")
print(data.isnull().sum())
# เลือกเฉพาะคอลัมน์ที่เป็นตัวเลข
numeric_columns = data.select_dtypes(include=['float64', 'int64']).columns
# เติมค่า NaN ในคอลัมน์ที่เป็นตัวเลขด้วยค่าเฉลี่ย
data[numeric_columns] = data[numeric_columns].fillna(data[numeric_columns].mean())
# ตรวจสอบค่าที่หายไปหลังการเติมค่า
print("\nค่าที่หายไปในแต่ละคอลัมน์หลังการเติมค่า:")
print(data.isnull().sum())


#### สร้างกราฟแท่งแสดงจำนวนคลาสของ Criteria ####
criteria_counts = data['Criteria'].value_counts()
# ปริ้นจำนวนคลาสแต่ละคลาส
print(criteria_counts)
# กำหนดขนาดของกราฟ
plt.figure(figsize=(10, 6))
# สร้างกราฟแท่ง
sns.barplot(x=criteria_counts.index, y=criteria_counts.values, palette='viridis')
# กำหนดชื่อกราฟและแกนเป็นภาษาอังกฤษ
plt.title('Number of Classes in Criteria', fontsize=16)
plt.xlabel('Criteria', fontsize=14)
plt.ylabel('Count', fontsize=14)
# แสดงกราฟ
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


# แปลง Criteria เป็นตัวเลข
data['Criteria'] = data['Criteria'].map({
    'Very Good': 4,
    'Good': 3,
    'Normal': 2,
    'Bad': 1,
    'Very Bad': 0
})

# แยกฟีเจอร์และเป้าหมาย
X = data.drop('Criteria', axis=1)
y = data['Criteria']
# ตรวจสอบการกระจายของคลาส
print(data['Criteria'].value_counts())

# ปรับ SMOTE เพื่อต่อสู้กับการไม่สมดุลของคลาส
from imblearn.over_sampling import SMOTE

# ตั้งค่า k_neighbors เป็นค่าที่น้อยลงตามขนาดของคลาสที่น้อย
smote = SMOTE(random_state=42, k_neighbors=1)  # หรือ k_neighbors=2 หากจำเป็น
X_resampled, y_resampled = smote.fit_resample(X, y)

# แปลงกลับเป็นชื่อคลาสเดิม
y_resampled = y_resampled.map({
    4: 'Very Good',
    3: 'Good',
    2: 'Normal',
    1: 'Bad',
    0: 'Very Bad'
})

# สร้าง DataFrame ใหม่สำหรับข้อมูลที่ถูกสุ่มตัวอย่าง
data_resampled = pd.DataFrame(X_resampled, columns=X.columns)
data_resampled['Criteria'] = y_resampled

# ตรวจสอบการกระจายของคลาสในข้อมูลที่ถูกสุ่มตัวอย่าง
print(data_resampled['Criteria'].value_counts())

# ฟังก์ชันเพื่อสร้างกราฟการกระจายของคลาส
def plot_class_distribution(original_data, resampled_data):
    plt.figure(figsize=(12, 6))

    # กราฟข้อมูลที่ถูกสุ่มตัวอย่าง
    sns.countplot(data=resampled_data, x='Criteria', palette='viridis')
    plt.title('Resampled Class Distribution with SMOTE', fontsize=16)
    plt.xlabel('Criteria', fontsize=14)
    plt.ylabel('Count', fontsize=14)

    plt.tight_layout()
    plt.show()

# เรียกใช้ฟังก์ชันเพื่อแสดงกราฟ
plot_class_distribution(data, data_resampled)

# ปริ้นตัวอย่างข้อมูล 10 แถวจาก data_resampled
print("ตัวอย่างข้อมูลจาก DataFrame ที่ถูกสุ่มตัวอย่าง:")
print(data_resampled.sample(10))

# ปรับเลขทศนิยมใน DataFrame ให้แสดงเพียง 2 ตำแหน่ง
data_resampled.iloc[:, :-1] = data_resampled.iloc[:, :-1].round(2)

# ปริ้นตัวอย่างข้อมูลอีกครั้ง
print("ตัวอย่างข้อมูลหลังจากปรับเลขทศนิยม:")
print(data_resampled.sample(10))

# บันทึก DataFrame ที่ถูกสุ่มตัวอย่างเป็นไฟล์ CSV
# data_resampled.to_csv('C:/Users/asus/Desktop/minproject_datamining_real/data/marine_data_clean.csv', index=False, encoding='utf-8-sig')
# print("ไฟล์ marine_data_clean.csv ถูกบันทึกเรียบร้อยแล้ว")