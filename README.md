
# AI Resume Screening & Job Matching System

An AI-powered web application that compares PDF resumes with job descriptions using Natural Language Processing (NLP) and semantic similarity.

## Features

- Upload multiple PDF resumes
- Compare resumes against a job description
- Calculate semantic similarity scores
- Identify matching and missing skills
- Rank candidates using heuristic relevance scores
- Visualize results in an interactive dashboard
- Export analysis results to CSV

## Technologies Used

- Python
- Streamlit
- Sentence Transformers
- Scikit-learn
- PyMuPDF
- Pandas
- NumPy
- Plotly

## How to Run

1. Clone the repository.
2. Create and activate a Python virtual environment.
3. Install dependencies:

   pip install -r requirements.txt

4. Start the application:

   streamlit run app.py

5. Open http://localhost:8501

## How It Works

The application extracts text from uploaded PDF resumes and uses Sentence Transformers to calculate semantic similarity with a job description. It also compares extracted skills and generates a weighted relevance score.

## Limitations

- Relevance scores are heuristic, not hiring probabilities.
- Skill extraction uses predefined keyword matching.
- Scanned PDFs may not be supported.
- Results should be reviewed by a human.

## Purpose

A portfolio project demonstrating NLP, semantic similarity, PDF processing, and interactive AI application development.
