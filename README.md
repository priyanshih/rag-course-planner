# Course Planning Assistant (RAG-based)

## Overview
This project implements a Retrieval-Augmented Generation (RAG) system that helps students:
- Check prerequisites
- Determine eligibility
- Plan courses for upcoming terms

The system strictly uses catalog data and avoids hallucination.

---

## Tech Stack
- Python
- LangChain
- FAISS (Vector DB)
- HuggingFace Embeddings

---

## Architecture

1. Load course + program documents
2. Split into chunks
3. Generate embeddings
4. Store in FAISS
5. Retrieve relevant chunks
6. Apply reasoning logic

---

## Project Structure

rag-course-planner/
│
├── main.py
├── requirements.txt
├── README.md
├── evaluation.txt
├── writeup.pdf
├── .gitignore
│
└── data/
├── cs101.txt
├── cs201.txt
├── cs301.txt
├── program.txt
├── policy.txt
└── ...

---

## Dataset

- 25+ course files
- Program requirements file
- Academic structure

---

## Sources

| Source | URL | Description | Date Accessed |
|-------|-----|------------|--------------|
| NPTEL Course Catalog | https://nptel.ac.in/courses | Structure and syllabus of engineering courses | March 2026 |
| IIT Madras BS Program | https://study.iitm.ac.in/ds/ | Program structure and course planning reference | March 2026 |
| MIT Course Catalog | https://catalog.mit.edu | Course prerequisites and academic structure | March 2026 |

Note: The dataset used in this project is curated and structured based on patterns observed in these publicly available academic catalogs.

---

## Features

-  Prerequisite reasoning
-  Course eligibility checking
-  Course planning
-  Safe abstention
-  Clarifying questions
-  Citation-based answers

---

## Evaluation Summary

- Citation Coverage: ~100% 
- Eligibility Accuracy: High (manual verification)
- Abstention Accuracy: Correct for unknown queries

---

## Setup and Run

pip install requirements.txt

python main.py


---

## Example Queries

- What is the prerequisite for CS301?
- Can I take CS401 if I’ve done CS201?
- Suggest courses if I’ve done CS101
- What is the full prerequisite chain for CS451?

---

## Limitations

- No semester availablity data
- No instructor-specific rules
- Grade constraints simplified

---

## Future Improvements

- Add full academic policies
- Integrate LLM-based reasoning
- Build UI
