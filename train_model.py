from ucimlrepo import fetch_ucirepo
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.impute import SimpleImputer
import joblib

# Ambil dataset
heart_disease = fetch_ucirepo(id=45)

# Ambil fitur dan target
X = heart_disease.data.features
y = heart_disease.data.targets

# Ubah target menjadi biner: 0 = tidak sakit, 1 = sakit
y = y['num'].apply(lambda val: 1 if val > 0 else 0)

# Cek apakah ada NaN di fitur
print("Jumlah NaN per kolom sebelum imputasi:")
print(X.isna().sum())

# Tangani NaN dengan imputasi (mengisi NaN dengan rata-rata kolom)
imputer = SimpleImputer(strategy='mean')
X = imputer.fit_transform(X)

# Cek kembali apakah ada NaN setelah imputasi
print("Jumlah NaN per kolom setelah imputasi:")
print(pd.DataFrame(X).isna().sum())

# Split data menjadi data latih dan data uji
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Latih model Naive Bayes
model = GaussianNB()
model.fit(X_train, y_train)

# Prediksi dengan data uji
y_pred = model.predict(X_test)

# Evaluasi model
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
print("\nAkurasi:", accuracy_score(y_test, y_pred))

# Simpan model ke file
joblib.dump(model, 'model/model_heart.pkl')
