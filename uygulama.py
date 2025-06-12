import streamlit as st
import pandas as pd
import numpy as np
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

# MxMH anket verisini yükleyelim
mxmh = pd.read_csv('mxmh_survey_results.csv')

# Müzik türleri ile ruh hali arasındaki ilişkiyi inceleyelim
genre_cols = [col for col in mxmh.columns if 'Frequency [' in col]
target_cols = ['Anxiety', 'Depression', 'Insomnia', 'OCD']

# NaN değerlerini temizleyelim
mxmh = mxmh.dropna(subset=genre_cols + target_cols)

# Frekansları sayısal hale getirelim
for col in genre_cols:
    mxmh[col] = mxmh[col].map({'Never': 0, 'Rarely': 1, 'Sometimes': 2, 'Very frequently': 3})

# Türler için ortalama ruh hali skorlarını hesaplayalım
genre_means = {}
for genre in genre_cols:
    avg_scores = mxmh[mxmh[genre] > 0][target_cols].mean()
    genre_means[genre.replace('Frequency [','').replace(']','')] = avg_scores

# Ortalamaları DataFrame formatına çevirelim
impact_df = pd.DataFrame(genre_means).T

# KMeans ile türleri sağlık etkisi sınıflarına ayıralım
scaler = StandardScaler()
scaled_data = scaler.fit_transform(impact_df)

kmeans = KMeans(n_clusters=3, random_state=42)
impact_df['cluster'] = kmeans.fit_predict(scaled_data)

# Sağlık etkisi sınıflarını isimlendirelim
cluster_order = impact_df.groupby('cluster')['Anxiety'].mean().sort_values().index.tolist()
label_map = {cluster_order[0]: 'Calming', cluster_order[1]: 'Neutral', cluster_order[2]: 'Anxious'}

# Impact DataFrame'e sağlık etkisi sınıfını ekleyelim
impact_df['health_effect'] = impact_df['cluster'].map(label_map)

# Spotify şarkı verisini yükleyelim
df = pd.read_csv('dataset.csv')

# Şarkı türlerini normalize edelim
df['normalized_genre'] = df['track_genre'].str.capitalize()

# MxMH'deki türlere göre sağlık etkisini eşleştirelim
genre_effect = impact_df['health_effect'].to_dict()

# Spotify verisine sağlık etkisi ekleyelim
df['health_effect'] = df['normalized_genre'].map(genre_effect)

# NaN olan değerleri temizleyelim
df = df.dropna(subset=['health_effect'])

# Özellikler
features = ['valence', 'energy', 'danceability', 'acousticness', 'tempo', 'instrumentalness', 'loudness']
X = df[features]
y = df['health_effect']

# Eğitim ve test verisi olarak bölelim
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Naive Bayes modelini oluşturma
model = GaussianNB()
model.fit(X_train, y_train)

# Streamlit Başlangıç
st.title("Spotify Şarkılarının Sağlık Etkisi Tahmini")
st.write("Bir şarkı seçin ve özelliklerini gözlemleyin!")

# Dropdown ile şarkı seçme
selected_song = st.selectbox("Şarkı Seçin", df['track_name'].unique())

# Seçilen şarkıyı filtreleme
song_data = df[df['track_name'] == selected_song].iloc[0]

# Şarkı bilgilerini gösterme
st.write(f"**Şarkı Adı:** {song_data['track_name']}")
st.write(f"**Sanatçı:** {song_data['artists']}")
st.write(f"**Tür:** {song_data['track_genre']}")
st.write(f"**Valence (Pozitiflik):** {song_data['valence']}")
st.write(f"**Energy (Enerji):** {song_data['energy']}")
st.write(f"**Tempo (BPM):** {song_data['tempo']}")

# Model ile tahmin yapma
song_features = song_data[features].values.reshape(1, -1)
predicted_effect = model.predict(song_features)

# Tahmin sonucunu gösterme
st.write(f"**Tahmin Edilen Sağlık Etkisi:** {predicted_effect[0]}")

# Olasılıkları gösterme
probs = model.predict_proba(song_features)[0]
classes = model.classes_

# Olasılıkları bar grafiği ile görselleştirme
probs_df = pd.DataFrame(probs, index=classes, columns=["Probability"])
st.bar_chart(probs_df)