from flask import Flask, render_template, request
import pickle
from pypdf import PdfReader
import re

app = Flask(__name__)

# Load trained model and TF-IDF vectorizer
model = pickle.load(open("model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))


# Clean resume text
def clean_text(text):
    text = text.lower()
    text = re.sub(r'\n', ' ', text)
    text = re.sub(r'[^a-zA-Z0-9+#.\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


# Detect skills from resume
def detect_skills(text):
    skills = [
        "python",
        "java",
        "c++",
        "sql",
        "machine learning",
        "deep learning",
        "data analysis",
        "pandas",
        "numpy",
        "scikit-learn",
        "tensorflow",
        "flask",
        "django",
        "html",
        "css",
        "javascript",
        "react",
        "node.js",
        "wordpress",
        "photoshop",
        "illustrator",
        "adobe indesign",
        "adobe acrobat",
        "excel",
        "powerpoint",
        "microsoft word",
        "power bi",
        "tableau",
        "aws",
        "azure",
        "git",
        "docker"
    ]

    text = text.lower()
    detected = []

    for skill in skills:
        pattern = r'(?<!\w)' + re.escape(skill) + r'(?!\w)'

        if re.search(pattern, text):
            detected.append(skill)

    return detected


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():

    if "resume" not in request.files:
        return "No resume file was uploaded."

    file = request.files["resume"]

    if file.filename == "":
        return "No file was selected."

    if not file.filename.lower().endswith(".pdf"):
        return "Invalid file type. Please upload a PDF resume."

    try:
        reader = PdfReader(file)

        pdf_text = ""

        for page in reader.pages:
            text = page.extract_text()

            if text:
                pdf_text += text

        if not pdf_text.strip():
            return (
                "Could not extract text from this PDF. "
                "Please upload a text-based PDF resume."
            )

        cleaned_text = clean_text(pdf_text)

        # Convert resume text into TF-IDF features
        resume_tfidf = vectorizer.transform([cleaned_text])

        # Predict job category
        prediction = model.predict(resume_tfidf)[0]

        # Detect skills
        detected_skills = detect_skills(pdf_text)

        # Resume statistics
        word_count = len(cleaned_text.split())
        page_count = len(reader.pages)
        skill_count = len(detected_skills)

        return render_template(
            "result.html",
            prediction=prediction,
            skills=detected_skills,
            word_count=word_count,
            page_count=page_count,
            skill_count=skill_count
        )

    except Exception:
        return (
            "An error occurred while processing the resume. "
            "Please make sure the uploaded file is a valid PDF."
        )


if __name__ == "__main__":
    app.run(debug=True)