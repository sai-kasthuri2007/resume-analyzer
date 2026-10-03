# Resume Analyzer

A machine learning-based Resume Analyzer that predicts the most relevant job category from an uploaded PDF resume.

## Project Overview

The system uses Natural Language Processing (NLP) and Machine Learning to analyze resume text and classify it into one of 24 job categories.

The application allows a user to:

- Upload a resume in PDF format
- Extract text from the resume
- Analyze the resume using TF-IDF
- Predict the relevant job category
- Detect predefined technical and professional skills
- Display resume statistics such as word count and page count

## Technologies Used

- Python
- Flask
- Scikit-learn
- TF-IDF
- Linear Support Vector Machine (LinearSVC)
- pypdf
- HTML
- CSS

## Machine Learning Pipeline

The project follows these steps:

1. Resume PDF collection
2. Text extraction using pypdf
3. Text cleaning
4. Train-test split
5. TF-IDF feature extraction
6. Model training using LinearSVC
7. Model evaluation
8. Model and vectorizer serialization using Pickle
9. Flask web application
10. Resume prediction

## Dataset

The model was trained using the Kaggle Resume Dataset.

The dataset contains resumes belonging to 24 different job categories.

Categories include:

- ACCOUNTANT
- ADVOCATE
- AGRICULTURE
- ARTS
- AUTOMOBILE
- AVIATION
- BANKING
- BPO
- BUSINESS-DEVELOPMENT
- CHEF
- CONSULTANT
- CONSTRUCTION
- DESIGNER
- DIGITAL-MEDIA
- ENGINEERING
- FITNESS
- FINANCE
- HEALTHCARE
- HR
- INFORMATION-TECHNOLOGY
- PUBLIC-RELATIONS
- SALES
- TEACHER
- APPAREL

## Model Performance

The final LinearSVC model achieved approximately:

**67% accuracy on the test dataset.**

The dataset contains multiple resume categories with different numbers of samples, so performance varies between categories.

## Application Features

### 1. Resume Upload

Users can upload a PDF resume through the web interface.

### 2. Job Category Prediction

The trained machine learning model predicts the most relevant job category.

### 3. Skill Detection

The application searches the extracted resume text for predefined skills such as:

- Python
- Java
- SQL
- Machine Learning
- HTML
- CSS
- JavaScript
- Photoshop
- Illustrator
- Excel
- PowerPoint
- Git
- Docker
- AWS

### 4. Resume Statistics

The result page displays:

- Number of words
- Number of pages
- Number of detected skills

## Project Structure

```text
resume-analyzer/
│
├── app.py
├── model.pkl
├── vectorizer.pkl
├── requirements.txt
├── .gitignore
│
└── templates/
    ├── index.html
    └── result.html