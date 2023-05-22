# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import zscore
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.preprocessing import Normalizer
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_selection import SelectKBest, f_classif
data = pd.read_csv("GLB (1).csv")

print(data.describe())

data_clean = data.dropna()
data_clean.to_csv('clean_data.csv', index=False)

#fill Missing values
#umple toate valorile lipsa cu 0
data_filled = data.fillna(value=0)
data_filled.to_csv('data_filled.csv', index = False)

print(data.columns)

#BoxPlot
# Convertim coloana la tipul float
data['Land-Ocean: Global Means'] = pd.to_numeric(data['Land-Ocean: Global Means'], errors='coerce')

plt.boxplot(data['Land-Ocean: Global Means'])
#plt.show()

# calculăm z-score
z_scores = zscore(data['Land-Ocean: Global Means'])
# creăm o nouă coloană pentru z-score
data['Z_scores'] = z_scores
# afișăm rândurile unde z-score este mai mare ca 3 sau mai mic ca -3
outliers = data[(data['Z_scores'] > 3) | (data['Z_scores'] < -3)]
print(outliers)

# Calcularea IQR
Q1 = data['Land-Ocean: Global Means'].quantile(0.25)
Q3 = data['Land-Ocean: Global Means'].quantile(0.75)
IQR = Q3 - Q1

lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

# Detectarea outlierilor
outliers2 = data[(data['Land-Ocean: Global Means'] < lower_bound) | (data['Land-Ocean: Global Means'] > upper_bound)]
print(outliers2)

#HISTORIGRAMA
# asigurați-vă că datele sunt curățate și pregătite înainte de a genera histograma
cleaned_data = data['Land-Ocean: Global Means'].dropna()

# generarea histograma
plt.hist(cleaned_data, bins=30, edgecolor='black')
plt.title('Histogram of Global Temperature Anomalies')
plt.xlabel('Temperature Anomalies (deg C)')
plt.ylabel('Frequency')
#plt.show()

#HEATMAP
# calculăm matricea de corelație
corr = data.corr()
# generăm heatmap
sns.heatmap(corr, xticklabels=corr.columns, yticklabels=corr.columns, cmap='RdBu')
plt.show()

#Standardization
# cream un scaler
scaler = StandardScaler()
# fit si transform pe date
data['Land-Ocean: Global Means'] = scaler.fit_transform(data['Land-Ocean: Global Means'].values.reshape(-1,1))

# cream un scaler
scaler = MinMaxScaler()
# fit si transform pe date
data['Land-Ocean: Global Means'] = scaler.fit_transform(data['Land-Ocean: Global Means'].values.reshape(-1,1))

#Normalization
# Înlocuirea valorilor NaN cu mediana coloanei
data['Land-Ocean: Global Means'].fillna((data['Land-Ocean: Global Means'].median()), inplace=True)
# Crearea unui normalizator L2
l2_normalizer = Normalizer(norm='l2')
# Aplicarea normalizării L2
data['Land-Ocean: Global Means'] = l2_normalizer.fit_transform(data['Land-Ocean: Global Means'].values.reshape(-1,1))

#Encoding categorical features
#OneHot Encoding
data = pd.get_dummies(data, columns=['Land-Ocean: Global Means'])
