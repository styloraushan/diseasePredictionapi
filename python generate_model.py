# generate_model.py
import pandas as pd
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder

DATA_PATH = "Dataset/diseasesymp_updated.csv"

df = pd.read_csv(DATA_PATH, encoding='latin1')
X = df.drop(columns=['label_dis'])
y = df['label_dis']

encoder = LabelEncoder()
y_encoded = encoder.fit_transform(y)

model = LogisticRegression(max_iter=200)
model.fit(X, y_encoded)

joblib.dump(model, 'model.pkl')
joblib.dump(encoder, 'label_encoder.pkl')

print("✅ model.pkl and label_encoder.pkl generated successfully.")
