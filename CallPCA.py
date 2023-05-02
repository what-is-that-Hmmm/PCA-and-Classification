import pandas
from sklearn.decomposition import PCA
import numpy as np
import matplotlib.pyplot as plt
# Reconfirm that python 3.8.8 is the correct version.

# Load original dataset
F_data = pandas.read_csv('data.csv',usecols=["F1","F2","F3","F4","F5","F6","F7","F8","F9","F10","F11","F12","F13","F14","F15"])


# Normalize the data in column manners
F_Normalized = F_data

for column in F_data.columns:
    F_Normalized[column] = F_Normalized[column]  / F_Normalized[column].abs().max()
# Print normalized data
print("Normalized Data Below:")
print(F_Normalized)

# Apply PCA
pc_num = 2
PC_project = PCA(n_components=pc_num)
PC_project.fit(F_Normalized)
print(PC_project.explained_variance_ratio_)

# View results of PCA
print("components are:")
print(PC_project.components_.T)
plt.plot(PC_project.explained_variance_ratio_, marker='o')
plt.ylabel('Explained Variance')
plt.xlabel('Components')
plt.show()

