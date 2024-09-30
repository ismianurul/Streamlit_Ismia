# Import libraries
import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# Page configuration
st.set_page_config(
    page_title="Outlier Detection and KMeans Clustering",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load dataset
@st.cache_data
def load_data():
    return pd.read_csv("data.csv")

dataset = load_data()

# Sidebar Configuration
st.sidebar.title('📊 Outlier Detection and KMeans Clustering')
st.sidebar.markdown('Gunakan sidebar untuk memilih parameter KMeans')

# Menampilkan 10 baris teratas dari dataset
st.subheader("Sample Data")
st.dataframe(dataset.head(10))

# Statistik dasar dataset
st.subheader("Dataset Statistics")
st.write(dataset.describe())

# Cek Missing Values
st.subheader("Missing Values")
st.write(dataset.isnull().sum())

# Cek tipe data
st.subheader("Data Types")
st.write(dataset.dtypes)

# Menentukan batas bawah dan atas untuk outlier
Q1 = dataset.select_dtypes(exclude=['object']).quantile(0.25)
Q3 = dataset.select_dtypes(exclude=['object']).quantile(0.75)
IQR = Q3 - Q1

batas_bawah = Q1 - 1.5 * IQR
batas_atas = Q3 + 1.5 * IQR

# Pilih kolom numerik dari dataframe
numeric_cols = dataset.select_dtypes(exclude=['object'])

# Membuat filter untuk mendeteksi outliers
outlier_filter = (numeric_cols < batas_bawah) | (numeric_cols > batas_atas)

# Menampilkan data yang terdeteksi sebagai outlier
outliers = dataset[outlier_filter.any(axis=1)]
st.subheader("Detected Outliers")
st.dataframe(outliers)

# Fitur Selection
X = dataset.select_dtypes(exclude=['object']).values

# Model K-means
# Definisikan list kosong untuk menyimpan WCSS (Within-Cluster Sum of Squares)
wcss = []

# Diasumsikan bahwa jumlah maksimal cluster yang mungkin ada dalam dataset adalah 10
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, init='k-means++', random_state=0)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)

# Visualisasi ELBOW
st.subheader("Elbow Method")
plt.figure(figsize=(10, 5))
plt.plot(range(1, len(wcss) + 1), wcss, marker='o', linestyle='--')
plt.title('The Elbow Method')
plt.xlabel('Number of Clusters')
plt.ylabel('WCSS')
st.pyplot(plt)

# Model Build
n_clusters = 5  # Misalkan 5 cluster berdasarkan analisis elbow
kmeansmodel = KMeans(n_clusters=n_clusters, init='k-means++', random_state=0)
y_kmeans = kmeansmodel.fit_predict(X)

# Visualisasi semua clusters
st.subheader("Clusters Visualization")
plt.figure(figsize=(10, 6))
colors = ['red', 'blue', 'green', 'cyan', 'magenta']
for i in range(n_clusters):
    plt.scatter(X[y_kmeans == i, 0], X[y_kmeans == i, 1], s=100, c=colors[i], label=f'Cluster {i + 1}')
plt.scatter(kmeansmodel.cluster_centers_[:, 0], kmeansmodel.cluster_centers_[:, 1], s=300, c='yellow', label='Centroids')
plt.title('Clusters of Customers')
plt.xlabel('Feature 1')  # Ganti dengan nama kolom yang sesuai
plt.ylabel('Feature 2')  # Ganti dengan nama kolom yang sesuai
plt.legend()
st.pyplot(plt)

# Visualisasi Outlier - Boxplot
st.subheader("Outlier Detection - Boxplot")
for column in dataset.select_dtypes(exclude=['object']):
    st.markdown(f'**{column}**')
    plt.figure(figsize=(10, 1.5))
    sns.boxplot(data=dataset, x=column)
    st.pyplot(plt)

# Footer
st.sidebar.markdown("### About")
st.sidebar.info(''' 
    Aplikasi ini menggunakan KMeans untuk mendeteksi outlier dan melakukan segmentasi berdasarkan data.
    Dataset diambil dari: pastikan untuk mengganti dengan dataset Anda.
''')
