import streamlit as st
import warnings
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
from function import train_model, predict

def app(df, x, y):
  st.title("Prediksi Penyakit Jantung")
  st.write("Masukkan data anda")
  
  col1, col2, col3 = st.columns(3)
  #input these features df[['Age', 'Sex', 'ChestPainType', 'RestingBP', 'Cholesterol', 'FastingBS', 'RestingECG', 'MaxHR', 'ExerciseAngina', 'Oldpeak', 'ST_Slope']]
  with col1:
    age = st.number_input("Umur", 1, 100)
    sex = st.selectbox("Jenis Kelamin", ['Male', 'Female'])
    chest_pain_type = st.selectbox("Tipe Sakit Dada", ['Asymptomatic', 'Atypical Angina', 'Non-anginal Pain', 'Typical Angina'])
    resting_bp = st.number_input("Tekanan Darah", 1, 200)
  with col2:
    cholesterol = st.number_input("Kolesterol", 1, 600)
    fasting_bs = st.selectbox("Gula Darah", [0,1])
    resting_ecg = st.selectbox("EKG", [ 'Left Ventricular Hypertrophy', 'Normal', 'ST-T Abnormality'])
    max_hr = st.number_input("Detak Jantung Maksimal", 1, 300)
  with col3:
    exercise_angina = st.selectbox("Angina Latihan", ['No', 'Yes'])
    oldpeak = st.number_input("Oldpeak", 1, 10)
    st_slope = st.selectbox("ST Slope", ['Downsloping', 'Flat', 'Upsloping'])

  if sex == "Male":
    sex = 1
  else:
    sex = 0

  if chest_pain_type == "Asymptomatic":
    chest_pain_type = 0
  elif chest_pain_type == "Atypical Angina":
    chest_pain_type = 1
  elif chest_pain_type == "Non-anginal Pain":
    chest_pain_type = 2
  else:
    chest_pain_type = 3

  if resting_ecg == "Left Ventricular Hypertrophy":
    resting_ecg = 0
  elif resting_ecg == "Normal":
    resting_ecg = 1
  else:
    resting_ecg = 2

  if exercise_angina == "No":
    exercise_angina = 0
  else:
    exercise_angina = 1

  if st_slope == "Downsloping":
    st_slope = 0
  elif st_slope == "Flat":
    st_slope = 1
  else:
    st_slope = 2

  features = [age, sex, chest_pain_type, resting_bp, cholesterol, fasting_bs, resting_ecg, max_hr, exercise_angina, oldpeak, st_slope]


  if st.button("Predict"):
    pred, score = predict(x, y, features)

    if pred == 0:
      st.success("Anda tidak memiliki penyakit jantung")
    else:
      st.error("Anda memiliki penyakit jantung")

    st.write("Akurasi: ", (score*100), "%")

  warnings.filterwarnings("ignore")
  st.set_option('deprecation.showPyplotGlobalUse', False)

  st.title("Visualisasi Data")

  if st.checkbox("Plot Confusion Matrix"):
    model, score = train_model(x,y)
    plt.figure(figsize=(10,10))
    pred = model.predict(x)
    cm = confusion_matrix(y, pred, labels=model.classes_)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                              display_labels=model.classes_)
    disp.plot()
    st.pyplot()
  
  if st.checkbox("Plot Scatter KNN"):
    model, score = train_model(x,y)
    plt.figure(figsize=(10,10))
    sns.scatterplot(x="Age", y="Cholesterol", hue="HeartDisease", data=df)
    plt.xlabel("Age")
    plt.ylabel("Cholesterol")
    plt.legend(loc='upper left')
    st.pyplot()


