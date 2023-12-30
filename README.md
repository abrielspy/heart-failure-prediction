# Laporan Proyek Machine Learning

### Nama : Abriel Salsabina P.Y

### Nim : 211351001

### Kelas : IF Pagi B

## Domain Proyek
Penyakit jantung merupakan salah satu penyebab kematian terbesar di dunia, dikarenakan pola hidup dan pola makan manusia yang semakin memburuk. Web App ini diciptakan untuk memudahkan ahli medis profesional dalam mengambil keputusan untuk pasiennya.

## Business Understanding
Ahli professional bisa menggunakan web app ini untuk memudahkan dan mempercepat proses diagnosa pasien, dengan itu maka akan semakin banyak manusia yang berhasil diperiksa dan diberikan penanganan yang tepat oleh ahli medis.

### Problem Statement
Semakin banyaknya manusia yang memiliki keluhan yang berkaitan dengan penyakit jantung dan ditambah dengan sedikitnya ahli medis pada bagian Cardiovaskular, maka mereka bisa dengan mudah merasa kewalahan dengan jumlah pasien yang bertambah.

### Goals
Bisa memudahkan dan meningkatkan kinerja ahli medis terutama seorang cardiologist.

### Solution Statements
- Membuat web app yang bisa memprediksi apakah pasien memiliki penyakit jantung atau tidak.

## Data Understanding
Dataset ini terciptakan karena banyaknya kasus penyakit jantung yang terjadi di dunia, data ini memiliki 1190 baris data namun 272 baris data merupakan data duplicate, maka data rilnya adalah 918 bari data dengan 12 kolom. Anda bisa mengakses datasetnya melalui link dibawah.
[Heart Failure Prediction](https://www.kaggle.com/datasets/fedesoriano/heart-failure-prediction)

### Variabel-variabel pada Diabetes Prediction adalah sebagai berikut:

- Age : Menunjukkan umur pasien [int, lebih dari 0]
- Sex : Menunjukkan jenis kelamin pasien [M: Male, F: Female]
- ChestPainType : Menunjukkan jenis sakit dada [TA : Typical Angina, ATA: Atypical Angina, NAP: Non-Anginal Pain, ASY: Asymptomatic]
- RestingBP : Menunjukkan tekanan darah saat tenang [mm Hg]
- Cholesterol : Menunjukkan jumlah cholesterol [mm/dl]
- FastingBS : Menunjukkan gula darah saat puasa [1: jika lebih dari 120, 0: jika tidak]
- RestingECG : Menunjukkan hasil electrocardiogram saat tenang [Normal: Normal, ST: memiliki keanehan ST-T, LVH: Memiliki hipertrofi ventrikel kiri]
- MaxHR : Menunjukkan jumlah detak jatung yang dihasilkan [int, lebih dari 0]
- ExerciseAngina : Menunjukkan angina akibat olahraga [Y: Yes, N: No]
- Oldpeak : Menunjukkan oldpeak sebelumnya [float]
- ST_Slope : Menunjukkan kemiringan puncak latihan ST [Up: upsloping, Flat: flat, Down: downsloping]
- HeartDisease : Menunjukkan apakah pasien memiliki penyakit jantung atau tidak[1: heart disease, 0: normal]

## Data Preparation
### Import Dataset
Seperti biasa langkah awal adalah memgimport file token kaggle untuk mendapatkan akses datasets dari kaggle.com.
```python
from google.colab import files
files.upload()
```
Dilanjut dengan membuat folder bagi file tersebut
```python
!mkdir -p ~/.kaggle
!cp kaggle.json ~/.kaggle/
!chmod 600 ~/.kaggle/kaggle.json
!ls ~/.kaggle
```
Mengunduh datasets yang kita inginkan dengan perintah dibawah
```python
!kaggle datasets download -d fedesoriano/heart-failure-prediction
```
Extract datasetsnya lalu masukkan ke dalam sebuah folder.
```python
!unzip heart-failure-prediction.zip -d data
!ls data
```
### Import library yang diperlukan
```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder
from sklearn import model_selection
from sklearn.neighbors import KNeighborsClassifier
from mlxtend.plotting import plot_decision_regions
import pickle
```
### Data Discovery & EDA
Langkah pertama pada tahap ini adalah membaca datasets yang tadi telah diextract.
```python
df2 = pd.read_csv('data/heart.csv')
df2.head(20)
```
Disini kita melihat apakah terdapat nilai null/NaN pada datasetsnya.
```python
df2.isnull().sum()
```
Tidak terdapat nilai null yaa.
```python
df2.nunique()
```
Diatas merupakan jumlah nilai unique pada kolom-kolom datasets, bisa dilihat terdapat beberapa kolom yang hanya memiliki 2 nilai unik, itu biasanya mengindikasikan true/false atau benar/salah. Mari kita cek apakah datasetnya memiliki nilai duplicate.
```python
df2.duplicated().sum()
```
Aman ya, 0 duplicate data.
```python
df2.describe()
```
Kita bisa lihat bahwa datasetnya memiliki 918 baris data sebelum diproses, namun nilai min bagi restingbp dan cholesterol terlalu rendah. Tidak mungkin bukan restingbp(heartbeat) bisa mencapai 0?? itu artinya pasien tersebut meninggal :") dan cholesterol 0 itu juga sangat mustahil karena setidaknya manusia membutuhkan kolesterol baik untuk hidup. Mari lanjut pada bagian EDA, kita akan memperbaiki nilai diatas pada tahap pre processing. <br>
Okeh, pertama-pertama kita akan melihat jumlah orang yang memiliki penyakit jantung dan yang tidak memiliki penyakit jantung.
```python
sns.countplot(data=df2, x='HeartDisease')
```
![download](https://github.com/abrielspy/heart-failure-prediction/assets/149224844/c32b2ddc-5369-4906-905f-2609f92ca8aa) <br>
hmm...terlihat cukup balance ya, tidak terlalu condong ke satu sisi, seharusnya aman dan tidak perlu kita lakukan tahap sampling (menambah/mengurangi data).
```python
plt.hist(df2[df2['Sex'] == 'M']['Age'], bins=10, alpha=0.9, label='Male')
plt.hist(df2[df2['Sex'] == 'F']['Age'], bins=10, alpha=0.9, label='Female')
plt.xlabel('Age')
plt.ylabel('Frequency')
plt.legend()
plt.show()
```
![download](https://github.com/abrielspy/heart-failure-prediction/assets/149224844/fa119148-ea75-4eab-bf96-de5f0af8d7d9)<br>
Dari plot diatas bisa kita lihat bahwa laki-laki lebih cenderung terkena penyakit jantung terutama pada saat berada diusia dewasa-lansia. Sedangkan wanita memiliki nilai yang lebih rendah. <br>
Kita akan membuat variable untuk menampung jumlah org yang mengidap kesakitan dada berdasarkan kategorinya.
```python
chest_count = df2.ChestPainType.value_counts()
plt.figure(figsize=(16, 8))
plt.pie(chest_count, labels=chest_count.index, autopct='%1.2f%%', startangle=100)
plt.show()
```
![download](https://github.com/abrielspy/heart-failure-prediction/assets/149224844/982bb02b-d592-4abc-bc44-78c64604aad0)<br>
Bisa dilihat bahwa mayoritas orang mengidap rasa sakit dada bertype ASY atau asymptomatic chestpain.
```python
sns.scatterplot(
    x="RestingBP",
    y="MaxHR",
    hue="HeartDisease",
    size="Age",
    data=df2,
    sizes=(20, 100),
    palette="coolwarm",
)
plt.show()
```
![download](https://github.com/abrielspy/heart-failure-prediction/assets/149224844/041ce984-1a3d-4bc4-9e96-024960bf80fe)<br>
Diatas menunjukkan korelasi antara RestingBP, MaxHR dan Age dimana umur yang rendah memiliki bulatan yang lebih kecil. hmm, ada satu data nyasar ya di nilai 0, jikalau kita bisa melakukan preprocessing terlebih dahulu mungkin kita bisa menunjukkan grafik yang lebih detail.
```
sns.boxplot(data=df2,x=df2.MaxHR,y=df2.Sex)
```
![download](https://github.com/abrielspy/heart-failure-prediction/assets/149224844/0a1298af-3bda-4b66-adc1-f46ca6933d3f)<br>
Disini menunjukkan heart rate wanita jauh lebih tinggi dibandingkan pria. Mungkinkah heartrate ini juga memiliki pengaruh yang besar dalam menentukan seseorang memiliki penyakit jantung atau tidak(?). Kita akan lanjut dengan tahap preprocessing.

### Data Cleansing & Pre-processing
Kita akan masukkan Label Encoder untuk mengubah data kategorial menjadi numerical.
```
le = LabelEncoder()
```
Karena tadi terdapat nilai 0 pada Cholesterol dan RestingBP, mari kita hilangkan data tersebut. Dengan mengetahui nilai mean dari masing-masing mean, kita bisa memfilter nilai yang abnormal dalam datasets.
```python
Q1 = df2[["Cholesterol", "RestingBP"]].quantile(0.25)
Q3 = df2[["Cholesterol", "RestingBP"]].quantile(0.75)
IQR = Q3 - Q1

mean_cholesterol = df2["Cholesterol"].mean()
mean_RestingBP = df2["RestingBP"].mean()

# Replace outliers with mean values in each column
df_cop = df2.copy()  # Create a copy to avoid modifying the original data
df_filtered = df_cop[~((df_cop["Cholesterol"] < (Q1["Cholesterol"] - 1.5 * IQR["Cholesterol"])) |
                     (df_cop["Cholesterol"] > (Q3["Cholesterol"] + 1.5 * IQR["Cholesterol"])) |
                     (df_cop["RestingBP"] < (Q1["RestingBP"] - 1.5 * IQR["RestingBP"])) |
                     (df_cop["RestingBP"] > (Q3["RestingBP"] + 1.5 * IQR["RestingBP"])))]
```
Kita memasukkan hasil dataframenya pada df_filtered
```
df_filtered["Sex"]=le.fit_transform(df_filtered["Sex"])
df_filtered["ChestPainType"]=le.fit_transform(df_filtered["ChestPainType"])
df_filtered["RestingECG"]=le.fit_transform(df_filtered["RestingECG"])
df_filtered["ExerciseAngina"]=le.fit_transform(df_filtered["ExerciseAngina"])
df_filtered["ST_Slope"]=le.fit_transform(df_filtered["ST_Slope"])
df_filtered.head()
```
Diatas merupakan hasil dari Label Encoder, dimana nilai unik kategorial diubah menjadi nomerical biasanya dimulai dari angka 0 hingga n.
```
df_filtered.to_csv('data.csv', index=False)
x=df_filtered.drop(columns='HeartDisease')
y=df_filtered.HeartDisease
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size = 0.3)
```
Membuat data train dengan 30% data test dan 70% data train.
## Modeling
Kita lanjut dengan process modeling, disini saya menggunakan n 23 :") setelah mencoba berkali-kali untuk mencari persentase yang bagus, ketemulah angka 23 ini.
```python
knn = KNeighborsClassifier(n_neighbors=23)
knn.fit(X_train, y_train)
y_pred = knn.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy}")
```
Hasil percentasenya adalah 71%
```python
test = [[40, 1, 1, 140, 289, 0, 1, 172, 0, 0.0, 2]]
test2 = [[49, 0, 2, 160, 180, 0, 1, 98, 0, 1.5, 1]]
pred = knn.predict(test2)
print(knn.predict(test2))
if pred == [0]:
  print('tidak memiliki penyakit jantung')
elif pred == [1]:
  print('memiliki penyakit jantung')
```
Data test diatas cukup akurat!
### Visualisasi hasil algoritma
```
k = 23  # Set the desired number of neighbors

# Iterate through unique pairs of features
feature_pairs = [
    ("MaxHR", "Cholesterol"),
    ("MaxHR", "RestingBP"),
    ("MaxHR", "Age"),
    ("Cholesterol", "RestingBP"),
    ("Cholesterol", "Age"),
    ("RestingBP", "Age"),
]



for x_column, y_column in feature_pairs:
    X = df_filtered[[x_column, y_column]].values
    y = df_filtered['HeartDisease'].astype(int).values

    # Train the KNN model
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X, y)

    # Create the decision region plot
    plot_decision_regions(X, y, clf=knn, legend=2)

    # Add axis labels and title
    plt.xlabel(x_column)
    plt.ylabel(y_column)
    plt.title(f'KNN Decision Regions for {x_column} vs. {y_column} (K={k})')

    plt.show()
```
![download](https://github.com/abrielspy/heart-failure-prediction/assets/149224844/45e3ebf8-91d9-442f-a7c9-8dbc91f6e59c)

## Evaluation
Untuk tahap evaluasi disini saya menggunakan precision, recall, f1-score serta confusion matrix. Saya menggunakan 4 tool ini karena ianya sangat cocok untuk mengevaluasi algorithma yang digunakan untuk melakukan klasifikasi.
```
print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))
```
Bisa dilihat hampir semua score precision, recall, dan f1-score adalah 71% dan confusion matrixnya juga masih masuk akal, dengan tingkat akurat 71%.

## Deployment
[Heart Failure Prediction](https://heart-failure-prediction-abriel.streamlit.app/) <br>
![image](https://github.com/abrielspy/heart-failure-prediction/assets/149224844/e24ed128-65ee-4cb2-b689-91c44c511619)

