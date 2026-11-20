
import streamlit as st
import fitz
import pandas as pd
import numpy as np
import plotly.express as px

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


# -------------------------------
# PAGE CONFIGURATION
# -------------------------------

st.set_page_config(
    page_title="AI Resume Screening System",
    page_icon="📄",
    layout="wide"
)

st.title("🤖 AI Resume Screening & Job Matching System")

st.write(
    "An NLP-powered application that compares candidate resumes "
    "with job descriptions using semantic similarity and skill analysis."
)


# -------------------------------
# LOAD AI MODEL
# -------------------------------

@st.cache_resource
def load_model():
    return SentenceTransformer("all-MiniLM-L6-v2")


with st.spinner("Loading NLP model..."):
    model = load_model()


# -------------------------------
# SKILL DATABASE
# -------------------------------

SKILLS = [
    "python", "java", "c++", "sql",
    "machine learning", "deep learning",
    "artificial intelligence", "data science",
    "natural language processing", "nlp",
    "computer vision", "tensorflow", "pytorch",
    "scikit-learn", "pandas", "numpy",
    "data analysis", "data visualization",
    "power bi", "tableau", "excel",
    "fastapi", "flask", "django",
    "streamlit", "react", "javascript",
    "html", "css", "docker",
    "git", "github", "aws", "azure",
    "generative ai", "large language models",
    "llm", "rag", "transformers",
    "opencv", "mediapipe", "statistics"
]


# -------------------------------
# PDF TEXT EXTRACTION
# -------------------------------

def extract_text(pdf_file):
    text = ""

    try:
        document = fitz.open(
            stream=pdf_file.getvalue(),
            filetype="pdf"
        )

        for page in document:
            text += page.get_text() + "\n"

        document.close()

    except Exception:
        return ""

    return text.strip()


# -------------------------------
# SKILL EXTRACTION
# -------------------------------

def extract_skills(text):
    text = text.lower()
    found = []

    for skill in SKILLS:
        if skill in text:
            found.append(skill)

    return sorted(set(found))


# -------------------------------
# SEMANTIC SIMILARITY
# -------------------------------

def calculate_similarity(resume_text, job_text):

    embeddings = model.encode(
        [resume_text, job_text],
        normalize_embeddings=True
    )

    similarity = cosine_similarity(
        [embeddings[0]],
        [embeddings[1]]
    )[0][0]

    return float(np.clip(similarity, 0, 1))


# -------------------------------
# JOB DESCRIPTION INPUT
# -------------------------------

st.sidebar.header("Job Requirements")

job_title = st.sidebar.text_input(
    "Job Title",
    value="AI/ML Engineer"
)

job_description = st.sidebar.text_area(
    "Paste Job Description",
    height=300,
    placeholder="Enter the complete job description here..."
)

uploaded_files = st.file_uploader(
    "Upload candidate resumes (PDF)",
    type=["pdf"],
    accept_multiple_files=True
)


# -------------------------------
# ANALYSIS
# -------------------------------

if st.button("🚀 Analyse Resumes", type="primary"):

    if not job_description.strip():
        st.error("Please enter a job description.")

    elif not uploaded_files:
        st.error("Please upload at least one PDF resume.")

    else:

        job_skills = extract_skills(job_description)

        results = []

        progress = st.progress(0)

        for index, file in enumerate(uploaded_files):

            resume_text = extract_text(file)

            if not resume_text:
                st.warning(
                    f"Could not extract text from {file.name}. "
                    "It may be scanned or password-protected."
                )
                progress.progress(
                    (index + 1) / len(uploaded_files)
                )
                continue

            resume_skills = extract_skills(resume_text)

            matching_skills = sorted(
                set(resume_skills) & set(job_skills)
            )

            missing_skills = sorted(
                set(job_skills) - set(resume_skills)
            )

            semantic_score = calculate_similarity(
                resume_text[:12000],
                job_description[:12000]
            )

            if job_skills:
                skill_score = (
                    len(matching_skills) / len(job_skills)
                )
            else:
                skill_score = 0.0

            # Weighted relevance score
            final_score = (
                0.70 * semantic_score +
                0.30 * skill_score
            ) * 100

            results.append({
                "Resume": file.name,
                "Semantic Similarity (%)": round(
                    semantic_score * 100, 2
                ),
                "Skill Coverage (%)": round(
                    skill_score * 100, 2
                ),
                "Relevance Score (%)": round(
                    final_score, 2
                ),
                "Matching Skills": ", ".join(
                    matching_skills
                ),
                "Skills to Review": ", ".join(
                    missing_skills
                ),
                "Extracted Skills": ", ".join(
                    resume_skills
                )
            })

            progress.progress(
                (index + 1) / len(uploaded_files)
            )

        if results:

            df = pd.DataFrame(results)

            df = df.sort_values(
                by="Relevance Score (%)",
                ascending=False
            ).reset_index(drop=True)

            df.insert(
                0,
                "Position in Results",
                range(1, len(df) + 1)
            )

            st.session_state["results"] = df
            st.session_state["job_title"] = job_title

        else:
            st.error(
                "No readable resumes were found. "
                "Try text-based PDF files."
            )


# -------------------------------
# RESULTS DASHBOARD
# -------------------------------

if "results" in st.session_state:

    df = st.session_state["results"]

    st.divider()

    st.header("📊 Candidate Analysis Dashboard")

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Resumes Analysed",
        len(df)
    )

    col2.metric(
        "Highest Relevance Score",
        f"{df['Relevance Score (%)'].max()}%"
    )

    col3.metric(
        "Average Relevance Score",
        f"{df['Relevance Score (%)'].mean():.2f}%"
    )

    st.subheader("Candidate Comparison")

    st.dataframe(
        df[
            [
                "Position in Results",
                "Resume",
                "Semantic Similarity (%)",
                "Skill Coverage (%)",
                "Relevance Score (%)"
            ]
        ],
        use_container_width=True,
        hide_index=True
    )

    st.subheader("Relevance Score Visualization")

    fig = px.bar(
        df,
        x="Resume",
        y="Relevance Score (%)",
        color="Relevance Score (%)",
        title="Resume Relevance Comparison",
        text="Relevance Score (%)"
    )

    fig.update_layout(
        xaxis_tickangle=-35,
        yaxis_range=[0, 100]
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    st.subheader("Detailed Candidate Analysis")

    selected_resume = st.selectbox(
        "Select a resume",
        df["Resume"].tolist()
    )

    candidate = df[
        df["Resume"] == selected_resume
    ].iloc[0]

    st.write(
        f"### {selected_resume}"
    )

    st.metric(
        "Relevance Score",
        f"{candidate['Relevance Score (%)']}%"
    )

    st.write("**Matching skills**")

    st.success(
        candidate["Matching Skills"] or "No matches detected"
    )

    st.write("**Skills to review**")

    st.warning(
        candidate["Skills to Review"] or "No gaps detected"
    )

    st.write("**All extracted skills**")

    st.write(
        candidate["Extracted Skills"] or "None detected"
    )

    csv = df.to_csv(index=False).encode("utf-8")

    st.download_button(
        "⬇️ Download Analysis Report",
        data=csv,
        file_name="resume_analysis_report.csv",
        mime="text/csv"
    )

    st.info(
        "This tool provides preliminary decision-support "
        "information, not hiring decisions. Scores are heuristic "
        "relevance indicators, not probabilities of job success. "
        "Recruiters should review resumes independently."
    )
