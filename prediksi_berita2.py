import joblib
from joblib import load
import streamlit as st
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from nltk.tokenize import word_tokenize
from nltk import download
from sklearn.feature_extraction.text import TfidfVectorizer
import json
import os

download('punkt')

savedFolder = "saved2"
rf_model = load(os.path.join(savedFolder, 'saved_model', 'rf_model.joblib'))
svm_model = load(os.path.join(savedFolder, 'saved_model', 'svm_model.joblib'))
Tfidf_vect = load(os.path.join(savedFolder, 'saved_model', 'Tfidf_vect.joblib'))

st.title('Prediksi Berita Hoax')

artikel_berita = st.text_input('Masukkan artikel berita')

predict = ''

factory = StemmerFactory()
stemmer = factory.create_stemmer()
# Tfidf_vect = TfidfVectorizer()

import re
def clean_text(text):
    # Remove special characters, newlines, and extra spaces
    cleaned_text = text.replace("\\n", " ")
    cleaned_text = cleaned_text.replace("\t", " ")
    cleaned_text = re.sub(r'\n\t+', ' ', cleaned_text)
    cleaned_text = re.sub(r'\s+', ' ', re.sub(r'[^A-Za-z\s]', '', cleaned_text))
    cleaned_text = cleaned_text.lower()
    return cleaned_text.strip()

def preprocess_input(input_text):
  lower = stemmer.stem(input_text.lower())
  tokens = word_tokenize(lower)
  return tokens

def transform_input(input_tokens):
  input_tfidf = Tfidf_vect.transform([" ".join(input_tokens)])
  return input_tfidf

def predict_svm(input_tfidf):
  return svm_model.predict(input_tfidf)

def predict_rf(input_tfidf):
  return rf_model.predict(input_tfidf)

def outputLabel(prediction):
  if prediction == 0:
    return "Berita Hoaks"
  elif prediction == 1:
    return "Bukan Berita Hoaks"

def score_akurasi(model="SVM"):
  with open(os.path.join(savedFolder, 'saved_predictions_accuracy_score', 'accuracy_score.json'), 'r') as jsonFile:
    jf = jsonFile.read()
  try:
    data = json.loads(jf)
  except json.JSONDecodeError as e:
    print(f"Error decoding JSON: {e}")
  accuracy_score = str(round(data[model]*100,2)) + "%"
  return accuracy_score

def manualTesting(artikel):
  cleaned_text =  clean_text(artikel)
  preprocessed_input = preprocess_input(cleaned_text)
  transformed_input = transform_input(preprocessed_input)
  predicted_svm = predict_svm(transformed_input[0])
  predicted_rf = predict_rf(transformed_input[0])
  return ("Prediksi dengan model SVM adalah " + outputLabel(predicted_svm) + " dengan skor akurasi: " + score_akurasi(model="SVM") + ".Prediksi dengan model Random Forest adalah " + outputLabel(predicted_rf) + " dengan skor akurasi: " + score_akurasi(model="Random Forest") + ".")

if st.button('Mulai Prediksi'):
  # test_artikel = ' - Wasabi adalah pasta hijau pedas yang sering disajikan dengan masakan Jepang, terutama sushi dan sashimi, untuk menambah rasa pada sausnya.Banyak yang menganggap wasabi sebagai makanan super karena kaya akan vitamin C dan juga memiliki sejumlah sifat antibakteri.Namun, mengapa wasabi sangat pedas? Dan apa saja bahan untuk membuat wasabi?Mungkin masih banyak dari kita yang bertanya-tanya, sebenarnya, wasabi terbuat dari apa?Jawaban sebenarnya adalah hanya ada satu bahan wasabi, yaitu wasabi. Untuk membuat wasabi, cukup dengan memarut rimpang tanaman Wasabia Japonica, yang langsung menjadi pasta wasabi yang siap dikonsumsi. Tidak ada campuran atau bahan lain yang digunakan.Namun, sebagian besar bumbu yang disajikan sebagai wasabi di restoran sebenarnya bukanlah wasabi asli.Wasabi palsu ini sebenarnya adalah lobak yang biasanya dicampur dengan mustard dan pewarna makanan hijau. Inilah sebabnya akar putih ini tampak hijau dan mungkin juga mengapa wasabi disalahartikan sebagai lobak pedas.Bahan pengental seperti tepung maizena serta bahan penstabil kimia juga umum digunakan untuk membuat wasabi palsu.Wasabi asli memiliki kepedasan dan rasa yang terkonsentrasi di dalam batang rimpangnya.Wasabi baru terasa pedas setelah batangnya diparut untuk menjadi pasta dan melepaskan rasa serta kepedasannya.Pemarutan memungkinkan fitokimia yang bertanggung jawab atas kepedasan wasabi bereaksi dengan udara karena sebagian besar tersebar di udara.Terkadang,brasa pedas wasabi asli terasa menusuk hidung saat diparut karena sifatnya yang mengudara.'
  # st.write(manualTesting(test_artikel)[0])
  # st.write(manualTesting(test_artikel)[1])
  if artikel_berita == "":
    st.write("Masukkan Artikel Berita")
  else:
    st.write(manualTesting(artikel_berita))