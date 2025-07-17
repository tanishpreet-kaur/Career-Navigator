import re
import spacy
import fitz  
import docx
import joblib
import numpy as np
import string


nlp = spacy.load("en_core_web_sm")

# Load DOCX
def read_docx(file_stream):
    doc = docx.Document(file_stream)
    return "\n".join([para.text for para in doc.paragraphs])

# Load PDF
def read_pdf(file_stream):
    text = ""
    with fitz.open(stream=file_stream.read(), filetype="pdf") as doc:
        for page in doc:
            text += page.get_text()
    return text

# Extract name 
def extract_name(text):
    lines = text.split('\n')
    lines = [line.strip() for line in lines if line.strip()]

    for line in lines[:10]:
        if line.isupper() and 2 <= len(line.split()) <= 4:
            if all(word.isalpha() for word in line.split()) and "RESUME" not in line:
                return line.title()

    doc = nlp(text)
    for ent in doc.ents:
        if ent.label_ == "PERSON" and ent.text.lower() not in SKILL_KEYWORDS:
            return ent.text.strip()

    email_match = re.search(r'([a-zA-Z0-9._%+-]+)@', text)
    if email_match:
        possible_name = email_match.group(1).replace('.', ' ').replace('_', ' ')
        return possible_name.title()

    return None


# Extract education 
EDUCATION_PATTERNS = {
    r'\b(b[\.\s-]*tech)\b': "B.Tech",
    r'\b(b[\.\s-]*e)\b': "B.E.",
    r'\b(m[\.\s-]*tech)\b': "M.Tech",
    r'\b(m[\.\s-]*e)\b': "M.E.",
    r'\b(ph[\.\s-]*d)\b': "Ph.D",
    r'\b(m[\.\s-]*sc)\b': "M.Sc",
    r'\b(b[\.\s-]*sc)\b': "B.Sc",
    r'\b(m[\.\s-]*ca)\b': "MCA",
    r'\b(b[\.\s-]*ca)\b': "BCA",
    r'\b(mba)\b': "MBA",
    r'\b(diploma)\b': "Diploma",
    r'\b(10th|secondary)\b': "Secondary",
    r'\b(12th|senior secondary)\b': "Senior Secondary",
}

def extract_education(text):
    text = text.lower()
    found = set()
    for pattern, label in EDUCATION_PATTERNS.items():
        if re.search(pattern, text):
            found.add(label)
    return sorted(list(found))


# Extract skills
SKILL_KEYWORDS = [
    "python", "java", "c++", "sql", "html", "css", "javascript", "excel",
    "flask", "django", "tensorflow", "pandas", "numpy", "machine learning",
    "deep learning", "data analysis", "git", "react", "node", "cloud"
]
def extract_skills(text):
    text = text.lower()
    found_skills = []
    for skill in SKILL_KEYWORDS:
        if re.search(r"\b" + re.escape(skill) + r"\b", text):
            found_skills.append(skill.title())
    return list(set(found_skills))

# ML model
model = joblib.load("job_role_model.joblib")
vectorizer = joblib.load("tfidf_vectorizer.joblib")

def clean_text(text):
    text = text.lower()
    text = text.replace('\n', ' ').replace('\r', ' ')
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = ''.join([ch for ch in text if not ch.isdigit()])
    return text

def predict_top_k_roles(text, k=3):
    cleaned = clean_text(text)
    vectorized = vectorizer.transform([cleaned])
    probabilities = model.predict_proba(vectorized)[0]
    top_k_indices = np.argsort(probabilities)[::-1][:k]
    top_k_roles = model.classes_[top_k_indices]
    top_k_probs = probabilities[top_k_indices]
    return list(zip(top_k_roles, top_k_probs))



# Main function
def parse_resume(file_stream, filename):
    if filename.endswith(".pdf"):
        text = read_pdf(file_stream)
    elif filename.endswith(".docx"):
        text = read_docx(file_stream)
    else:
        raise ValueError("Unsupported file type")

    skills = extract_skills(text)
    education = extract_education(text)
    top_roles = predict_top_k_roles(text, k=3)

    formatted_skills = ", ".join(skills)
    formatted_education = ", ".join(education)

    return {
        "name": extract_name(text),
        "education": formatted_education,      
        "skills": formatted_skills,            
        "top_job_roles": top_roles,      
        "raw_text": text[:1000]
    }




