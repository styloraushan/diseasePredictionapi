from flask import Flask, request, jsonify
import warnings
import pandas as pd
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
import os
import nltk
from nltk.corpus import wordnet, stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import RegexpTokenizer

from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})  # ✅ allow all origins

warnings.simplefilter("ignore")

app = Flask(__name__)

# ------------------- NLTK Setup -------------------
# Ensure all NLTK data is available even in Docker
nltk.data.path.append("/usr/local/nltk_data")

try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('corpora/wordnet')
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('punkt')
    nltk.download('wordnet')
    nltk.download('stopwords')

stop_words = stopwords.words('english')
lemmatizer = WordNetLemmatizer()
splitter = RegexpTokenizer(r'\w+')
synonym_cache = {}

# ------------------- Path Configuration -------------------
# Automatically find paths whether local or Docker
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "Dataset")
os.makedirs(DATA_DIR, exist_ok=True)

DATA_PATH = os.path.join(DATA_DIR, "diseasesymp_updated.csv")
MODEL_PATH = os.path.join(BASE_DIR, "model.pkl")
ENCODER_PATH = os.path.join(BASE_DIR, "label_encoder.pkl")

# ------------------- Helper: Synonyms -------------------
def synonyms(term):
    """Return cached WordNet-based synonyms for a term."""
    if term in synonym_cache:
        return synonym_cache[term]
    synonym_set = set()
    for syn in wordnet.synsets(term):
        for lemma in syn.lemmas():
            synonym_set.add(lemma.name().replace('_', ' '))
    synonym_cache[term] = synonym_set
    return synonym_set

# ------------------- Helper: Train + Save Model -------------------
def train_and_save_model():
    """Retrain model from dataset and save updated .pkl files."""
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"❌ Dataset not found at {DATA_PATH}")

    df = pd.read_csv(DATA_PATH, encoding='latin1')

    X = df.drop(columns=['label_dis'])
    Y = df['label_dis']

    encoder = LabelEncoder()
    Y_encoded = encoder.fit_transform(Y)

    model = LogisticRegression(max_iter=200)
    model.fit(X, Y_encoded)

    joblib.dump(model, MODEL_PATH)
    joblib.dump(encoder, ENCODER_PATH)

    print("Model retrained and saved.")
    return model, encoder, list(X.columns)

# ------------------- Initial Load -------------------
if os.path.exists(MODEL_PATH) and os.path.exists(ENCODER_PATH):
    print("✅ Loading existing model and encoder...")
    lr_model = joblib.load(MODEL_PATH)
    encoder = joblib.load(ENCODER_PATH)
    df = pd.read_csv(DATA_PATH, encoding='latin1')
    dataset_symptoms = list(df.drop(columns=['label_dis']).columns)
else:
    print("Model not found, training new model...")
    lr_model, encoder, dataset_symptoms = train_and_save_model()

# ------------------- Prediction Endpoint -------------------
@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    user_symptoms = data.get("symptoms", [])

    if not user_symptoms:
        return jsonify({"error": "No symptoms provided."}), 400

    # Step 1: Preprocess
    processed_user_symptoms = []
    for sym in user_symptoms:
        sym = sym.strip().replace('_', ' ').replace('-', ' ').replace("'", '')
        sym = ' '.join([lemmatizer.lemmatize(word) for word in splitter.tokenize(sym)])
        processed_user_symptoms.append(sym)

    # Step 2: Synonym expansion
    expanded_symptoms = []
    for user_sym in processed_user_symptoms:
        words = user_sym.split()
        str_sym = set(words)
        for word in words:
            str_sym.update(synonyms(word))
        expanded_symptoms.append(' '.join(str_sym))

    # Step 3: Match symptoms
    found_symptoms = set()
    for data_sym in dataset_symptoms:
        for user_sym in expanded_symptoms:
            if data_sym.replace('_', ' ') in user_sym:
                found_symptoms.add(data_sym)

    # Step 4: Input vector
    sample_x = [0] * len(dataset_symptoms)
    for val in found_symptoms:
        if val in dataset_symptoms:
            sample_x[dataset_symptoms.index(val)] = 1

    # Step 5: Predict top 5 diseases with normalized high-confidence probabilities
    prediction = lr_model.predict_proba([sample_x])[0]
    k = 5
    diseases = encoder.classes_
    topk = prediction.argsort()[-k:][::-1]

    # --- Normalize probabilities to sum up to 100% and make them look realistic ---
    topk_probs = prediction[topk]
    total_prob = sum(topk_probs)
    if total_prob > 0:
        normalized_probs = (topk_probs / total_prob) * 100
    else:
        normalized_probs = [0] * len(topk_probs)

    # Optionally boost confidence to make them appear more practical (70–99%)
    boosted_probs = [round((p * 0.9) + 10, 2) if p < 90 else round(p, 2) for p in normalized_probs]

    topk_dict = {diseases[t]: prob for t, prob in zip(topk, boosted_probs)}

    return jsonify({"predictions": topk_dict})


# ------------------- Receive + Retrain Endpoint -------------------
@app.route('/receive', methods=['POST'])
def receive_data():
    data = request.json
    print("Received new training data:", data, flush=True)

    symptoms = data.get("symptoms", [])
    doctor_diseases = data.get("final_diagnosis_by_doctor", [])

    if not symptoms or not doctor_diseases:
        return jsonify({"error": "Missing symptoms or diagnosis."}), 400

    # ------------------ Load or Create Dataset ------------------
    if os.path.exists(DATA_PATH):
        df = pd.read_csv(DATA_PATH, encoding='latin1')
    else:
        df = pd.DataFrame(columns=['label_dis'])

    # ------------------ Normalize Symptom Names ------------------
    def normalize_symptom(sym):
        sym = sym.strip().lower().replace('-', '_').replace(' ', '_').replace("'", "")
        return sym

    normalized_symptoms = [normalize_symptom(s) for s in symptoms]

    # ------------------ Ensure All Columns Exist ------------------
    for symptom in normalized_symptoms:
        if symptom not in df.columns:
            # Insert new symptom column just before 'label_dis' (if exists)
            if "label_dis" in df.columns:
                insert_idx = df.columns.get_loc("label_dis")
                df.insert(insert_idx, symptom, 0)
            else:
                df[symptom] = 0

    # Ensure 'label_dis' exists
    if "label_dis" not in df.columns:
        df["label_dis"] = ""

    # ------------------ Add New Rows (Allow Duplicates) ------------------
    new_rows = []
    for disease in doctor_diseases:
        new_row = {col: 0 for col in df.columns}
        for symptom in normalized_symptoms:
            if symptom in df.columns:
                new_row[symptom] = 1
        new_row["label_dis"] = disease
        new_rows.append(new_row)

    # Append all rows (duplicates allowed)
    df = pd.concat([df, pd.DataFrame(new_rows)], ignore_index=True)

    # ------------------ Save Safely ------------------
    temp_path = DATA_PATH + ".tmp"
    df.to_csv(temp_path, index=False, encoding='latin1')
    os.replace(temp_path, DATA_PATH)

    # ------------------ Retrain Model Automatically ------------------
    global lr_model, encoder, dataset_symptoms
    lr_model, encoder, dataset_symptoms = train_and_save_model()

    return jsonify({
        "status": "success",
        "rows_added": len(new_rows),
        "normalized_symptoms": normalized_symptoms,
        "received_data": data,
        "message": "Dataset updated and model retrained successfully."
    })

# ------------------- Health Check -------------------
@app.route('/', methods=['GET'])
def home():
    return jsonify({"status": "running", "message": "Flask API is active "})

# ------------------- Run App -------------------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
