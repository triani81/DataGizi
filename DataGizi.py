import streamlit as st
import pandas as pd
import seaborn as sns
import  matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler, Normalizer, LabelEncoder
import joblib

# Inisialisasi session state
if 'model' not in st.session_state:
    st.session_state.model = None
if 'normalizer' not in st.session_state:
    st.session_state.normalizer = None
if 'scaler' not in st.session_state:
    st.session_state.scaler = None
if 'le' not in st.session_state:
    st.session_state.le = None
if 'apply_normalization' not in st.session_state:
    st.session_state.apply_normalization = False
if 'apply_standardization' not in st.session_state:
    st.session_state.apply_standardization = False
if 'apply_label_encoding' not in st.session_state:
    st.session_state.apply_label_encoding = False
if 'feature_names' not in st.session_state:
    st.session_state.feature_names = None

# Judul Aplikasi
st.title("Aplikasi Klasifikasi Data Gizi dengan Naive Bayes")

# Pilih Halaman
page = st.sidebar.selectbox("Pilih Halaman", ["Upload Data", "Prediksi Manual"])

if page == "Upload Data":
    # Upload Dataset
    st.sidebar.header("Upload Dataset Anda")
    uploaded_file = st.sidebar.file_uploader("Pilih file CSV", type=["csv"])

    if uploaded_file is not None:
        # Membaca dataset
        data = pd.read_csv(uploaded_file)

        st.write("## Dataset")
        st.write(data)

        # Pilih Fitur dan Label
        st.sidebar.header("Pilih Fitur dan Label")
        all_columns = data.columns.tolist()
        features = st.sidebar.multiselect("Pilih Fitur", all_columns, default=all_columns[:-1])
        label = st.sidebar.selectbox("Pilih Label", all_columns, index=len(all_columns)-1)

        # Simpan nama fitur di session state
        st.session_state.feature_names = features

        # Pilih Preprocessing
        st.sidebar.header("Pilih Preprocessing")
        apply_normalization = st.sidebar.checkbox("Gunakan Normalizer")
        apply_standardization = st.sidebar.checkbox("Gunakan Standard Scaler")
        apply_label_encoding = st.sidebar.checkbox("Gunakan Label Encoder pada Label")

        # Simpan pilihan preprocessing di session state
        st.session_state.apply_normalization = apply_normalization
        st.session_state.apply_standardization = apply_standardization
        st.session_state.apply_label_encoding = apply_label_encoding

        if st.sidebar.button("Mulai Klasifikasi"):
            # Memisahkan Fitur dan Label
            X = data[features]
            y = data[label]

            # Mengatasi Kolom Kategorikal (Label Encoding)
            le = LabelEncoder()
            for column in X.columns:
                if X[column].dtype == 'object':
                    X[column] = le.fit_transform(X[column])

            # Preprocessing: Label Encoding (jika dipilih)
            if apply_label_encoding:
                y = le.fit_transform(y)

            # Preprocessing: Normalisasi (jika dipilih)
            if apply_normalization:
                normalizer = Normalizer()
                X = normalizer.fit_transform(X)
                st.session_state.normalizer = normalizer
            else:
                st.session_state.normalizer = None

            # Preprocessing: Standardisasi (jika dipilih)
            if apply_standardization:
                scaler = StandardScaler()
                X = scaler.fit_transform(X)
                st.session_state.scaler = scaler
            else:
                st.session_state.scaler = None

            # Bagi data menjadi training dan testing set
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            # Menampilkan Visualisasi Pembagian Data
            with st.container():
                st.write("## Visualisasi Pembagian Data")
                
                # Proporsi Data Training dan Testing
                train_size = len(X_train)
                test_size = len(X_test)
                
                # Pie Chart untuk Pembagian Data
                st.write("### Pie Chart: Pembagian Data Training dan Testing")
                sizes = [train_size, test_size]
                labels = ['Training Set', 'Testing Set']
                fig, ax = plt.subplots()
                ax.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=140, colors=['#ff9999','#66b3ff'])
                ax.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
                st.pyplot(fig)
                
                # Bar Chart untuk Pembagian Data
                st.write("### Bar Chart: Pembagian Data Training dan Testing")
                df_sizes = pd.DataFrame({
                    'Set': ['Training Set', 'Testing Set'],
                    'Size': [train_size, test_size]
                })
                fig, ax = plt.subplots()
                sns.barplot(x='Set', y='Size', data=df_sizes, palette='Set2', ax=ax)
                ax.set_title('Jumlah Data Training dan Testing')
                ax.set_xlabel('Set')
                ax.set_ylabel('Jumlah Data')
                st.pyplot(fig)

            # Membuat Model Naive Bayes
            model = GaussianNB()
            model.fit(X_train, y_train)

            # Simpan model dan preprocessing ke session state
            st.session_state.model = model
            st.session_state.le = le

            # Melakukan Prediksi
            y_pred = model.predict(X_test)

            # Menampilkan Hasil dan Visualisasi secara Keseluruhan
            with st.container():
                st.write("## Hasil Prediksi")
                st.write(f"Akurasi: **{accuracy_score(y_test, y_pred)*100:.2f}%**")
                st.write("### Laporan Klasifikasi")
                st.text(classification_report(y_test, y_pred))

                # Visualisasi: Matriks Kebingungan (Confusion Matrix)
                st.write("### Matriks Kebingungan")
                cm = confusion_matrix(y_test, y_pred)
                plt.figure(figsize=(8, 6))
                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=le.classes_, yticklabels=le.classes_)
                plt.title("Confusion Matrix")
                plt.xlabel("Prediksi")
                plt.ylabel("Aktual")
                st.pyplot(plt)

                # Visualisasi: Distribusi Kelas
                st.write("### Distribusi Kelas dalam Dataset")
                plt.figure(figsize=(8, 6))
                sns.countplot(x=y, palette='Set2')
                plt.title("Distribusi Kelas")
                plt.xlabel("Kelas")
                plt.ylabel("Jumlah")
                st.pyplot(plt)

                # Visualisasi: Distribusi Fitur
                st.write("### Distribusi Fitur")
                for feature in features:
                    plt.figure(figsize=(8, 6))
                    sns.histplot(data[feature], kde=True, color='skyblue')
                    plt.title(f"Distribusi {feature}")
                    plt.xlabel(feature)
                    plt.ylabel("Frekuensi")
                    st.pyplot(plt)

    else:
        st.write("Silakan unggah dataset CSV untuk melatih model.")

elif page == "Prediksi Manual":
    st.header("Prediksi Manual")

    if st.session_state.model is None:
        st.warning("Silakan latih model terlebih dahulu di halaman 'Upload Data'.")
    else:
        model = st.session_state.model
        normalizer = st.session_state.normalizer
        scaler = st.session_state.scaler
        le = st.session_state.le
        feature_names = st.session_state.feature_names
        
        # Input data manual
        usia = st.number_input("Masukkan Usia:", min_value=0, max_value=60)
        berat = st.number_input("Masukkan Berat Badan (kg):", min_value=0.0, max_value=100.0)
        tinggi = st.number_input("Masukkan Tinggi Badan (cm):", min_value=0.0, max_value=200.0)

        if st.button("Lakukan Prediksi"):
        # Data input manual disimpan dalam DataFrame dengan nama kolom yang sesuai
         input_data = pd.DataFrame([[usia, berat, tinggi]], columns=feature_names)

        # Preprocessing jika diperlukan
         if st.session_state.apply_normalization and normalizer is not None:
               input_data = normalizer.transform(input_data)

    if st.session_state.apply_standardization and scaler is not None:
        input_data = scaler.transform(input_data)

    # Prediksi
    prediction = model.predict(input_data)

    # Pastikan prediksi mengenali "Gizi Lebih"
    if le is not None:
        prediction = le.inverse_transform(prediction)
    
    # Tampilkan hasil prediksi
    if prediction[0] == 'Gizi Lebih':
        st.success(f"Hasil Prediksi: {prediction[0]} (Gizi Lebih)")
    else:
        st.write(f"Hasil Prediksi: {prediction[0]}")

