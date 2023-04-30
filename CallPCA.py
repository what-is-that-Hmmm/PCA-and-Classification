import pandas
from sklearn.decomposition import PCA
import numpy
import matplotlib.pyplot as plot
# Reconfirm that python 3.8.8 is the correct version.

# Load original dataset
F_data = pandas.read_csv('data.csv',usecols=["F1","F2","F3","F4","F5","F6","F7","F8","F9","F10","F11","F12","F13","F14","F15"])

# You must normalize the data before applying the fit method
F_Normalized = F_data
# Normalize the data in column manners
for column in F_data.columns:
    F_Normalized[column] = F_Normalized[column]  / F_Normalized[column].abs().max()
print("Normalized Data Below:")
print(F_Normalized)


pca = PCA(n_components=F_data.shape[1])
pca.fit(F_Normalized)

# Reformat
loadings = pandas.DataFrame(pca.components_.T,
columns=['PC%s' % _ for _ in range(len(F_Normalized.columns))],
index=F_data.columns)
print(loadings)

# View results
plot.plot(pca.explained_variance_ratio_)
plot.ylabel('Explained Variance')
plot.xlabel('Components')
plot.show()