from flask import Flask, request, jsonify
from flask_cors import CORS
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from sentence_transformers import SentenceTransformer, util
import re

# Inisialisasi Flask
app = Flask(__name__)
CORS(app)  # biar bisa dipanggil React

# Inisialisasi TF-IDF Vectorizer
vectorizer = TfidfVectorizer(stop_words="indonesian")

# Buat stemmer bahasa Indonesia
factory = StemmerFactory()
stemmer = factory.create_stemmer()

# Load model BERT
model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')


# ----------------- Fungsi Preprocessing -----------------
def preprocess(text):
    # lowercase
    text = text.lower()
    # hapus angka dan tanda baca
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    # stemming
    text = stemmer.stem(text)
    return text


# ----------------- Fungsi Similarity -----------------
def jaccard_similarity(text1, text2):
    set1 = set(text1.split())
    set2 = set(text2.split())
    return len(set1 & set2) / len(set1 | set2)


def bert_similarity(text1, text2):
    embeddings = model.encode([text1, text2], convert_to_tensor=True)
    similarity = util.cos_sim(embeddings[0], embeddings[1]).item()
    return similarity


# ----------------- Endpoint Flask -----------------
@app.route("/plagiarism-check", methods=["POST"])
def plagiarism_check():
    data = request.json
    text1 = preprocess(data.get("text1", ""))
    text2 = preprocess(data.get("text2", ""))

    if not text1 or not text2:
        return jsonify({"error": "Teks tidak boleh kosong"}), 400

    # TF-IDF + Cosine
    vectors = vectorizer.fit_transform([text1, text2])
    tfidf_similarity = cosine_similarity(vectors[0], vectors[1])[0][0]

    # Jaccard
    jaccard = jaccard_similarity(text1, text2)

    # BERT Semantic
    bert = bert_similarity(text1, text2)

    # Ubah semua ke persen
    tfidf_percent = round(tfidf_similarity * 100, 2)
    jaccard_percent = round(jaccard * 100, 2)
    bert_percent = round(bert * 100, 2)

    # Ambil rata-rata buat "final score"
    final_score = round((tfidf_percent + jaccard_percent + bert_percent) / 3, 2)

    # Status kategori
    if final_score >= 80:
        status = "Tinggi (kemungkinan besar plagiat)"
    elif final_score >= 50:
        status = "Sedang (ada kemiripan)"
    else:
        status = "Rendah (aman)"

    return jsonify({
        "tfidf_similarity": tfidf_percent,
        "jaccard_similarity": jaccard_percent,
        "bert_similarity": bert_percent,
        "final_score": final_score,
        "status": status
    })


# ----------------- Main -----------------
if __name__ == "__main__":
    app.run(debug=True)
