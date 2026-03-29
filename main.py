import os
import re
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

# -------------------------------
# LOAD DATA
# -------------------------------

documents = []
SOURCE_MAP = {}

for file in os.listdir("data"):
    if file.endswith(".txt"):
        loader = TextLoader(f"data/{file}")
        docs = loader.load()

        for doc in docs:
            doc.metadata["source"] = file

        documents.extend(docs)
        SOURCE_MAP[file] = f"https://example.edu/catalog/{file.replace('.txt','')}"

# -------------------------------
# RAG SETUP
# -------------------------------

splitter = CharacterTextSplitter(chunk_size=400, chunk_overlap=50)
chunks = splitter.split_documents(documents)

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

db = FAISS.from_documents(chunks, embeddings)
retriever = db.as_retriever(search_kwargs={"k": 5})

# -------------------------------
# BUILD PREREQ MAP
# -------------------------------

def build_prereq_map(docs):
    prereq_map = {}

    for doc in docs:
        lines = doc.page_content.split("\n")

        course = None
        prereq = None

        for line in lines:
            line = line.strip()

            match = re.match(r"([A-Za-z]{2,}\d{2,})", line)
            if match:
                course = match.group().upper()

            if "prerequisite" in line.lower():
                prereq = line.split(":")[-1].strip().upper()

        if course:
            prereq_map[course] = prereq if prereq else "NONE"

    return prereq_map

prereq_map = build_prereq_map(documents)

# -------------------------------
# HELPERS
# -------------------------------

def extract_courses(query):
    return list(set([c.upper() for c in re.findall(r"[A-Za-z]{2,}\d{2,}", query)]))

def expand_completed(completed):
    expanded = set(completed)

    for course in list(expanded):
        while course in prereq_map:
            prereq = prereq_map[course]

            if not prereq or prereq.lower() == "none":
                break

            if prereq in expanded:
                break

            expanded.add(prereq)
            course = prereq

    return expanded

def get_chain(course):
    chain = []
    visited = set()

    while True:
        if course not in prereq_map:
            break

        prereq = prereq_map[course]

        if not prereq or prereq.lower() == "none":
            break

        if prereq in visited:
            break

        chain.append(prereq)
        visited.add(prereq)
        course = prereq

    return chain

# -------------------------------
# QUERY TYPE DETECTION
# -------------------------------

def is_planning_query(q):
    q = q.lower()
    return any(k in q for k in [
        "suggest", "plan", "semester",
        "what courses can i take",
        "what can i take", "after"
    ])

def is_prereq_query(q):
    return any(k in q.lower() for k in ["prerequisite", "chain"])

def is_program_query(q):
    return any(k in q.lower() for k in [
        "credit", "degree", "requirement", "elective", "program"
    ])

def is_mandatory_query(q):
    return any(k in q.lower() for k in [
        "mandatory", "required", "compulsory", "do i need"
    ])

def is_policy_query(q):
    return any(k in q.lower() for k in [
        "skip", "override", "waive", "exception", "consent"
    ])

def is_reverse_query(q):
    return "require" in q.lower() and "what" in q.lower()

# -------------------------------
# CITATIONS
# -------------------------------

def get_citations(courses):
    citations = []

    for course in courses:
        filename = f"{course.lower()}.txt"
        url = SOURCE_MAP.get(filename, "")

        try:
            with open(f"data/{filename}", "r") as f:
                for line in f:
                    if "prerequisite" in line.lower():
                        citations.append(f"{url} | {line.strip()}")
                        break
        except:
            continue

    return "\n- " + "\n- ".join(citations) if citations else "\n- None"

# -------------------------------
# PROGRAM QA
# -------------------------------

def answer_program_question(query):
    docs = retriever.invoke(query)

    for doc in docs:
        for line in doc.page_content.split("\n"):
            if "credit" in line.lower():
                url = SOURCE_MAP.get(doc.metadata.get("source", ""), "")
                return f"""
Answer / Plan:
{line.strip()}

Why:
Found in program document

Citations:
- {url} | {line.strip()}

Clarifying questions:
- None

Assumptions / Not in catalog:
- None
"""
    return None

# -------------------------------
# MANDATORY QA
# -------------------------------

def answer_mandatory_question(query):
    docs = retriever.invoke(query)

    for doc in docs:
        for line in doc.page_content.split("\n"):
            if "required" in line.lower() or "mandatory" in line.lower():
                url = SOURCE_MAP.get(doc.metadata.get("source", ""), "")
                return f"""
Answer / Plan:
{line.strip()}

Why:
Found in program document

Citations:
- {url} | {line.strip()}

Clarifying questions:
- None

Assumptions / Not in catalog:
- None
"""
    return None

# -------------------------------
# POLICY QA
# -------------------------------

def answer_policy_question(query):
    docs = retriever.invoke(query)

    for doc in docs:
        for line in doc.page_content.split("\n"):
            if any(k in line.lower() for k in ["consent", "exception", "override"]):
                url = SOURCE_MAP.get(doc.metadata.get("source", ""), "")
                return f"""
Answer / Plan:
{line.strip()}

Why:
Found in policy document

Citations:
- {url} | {line.strip()}

Clarifying questions:
- None

Assumptions / Not in catalog:
- None
"""

    return """
Answer / Plan:
I don't have that information in the provided catalog.

Why:
Policy not defined

Citations:
- None

Clarifying questions:
- Check academic advisor

Assumptions / Not in catalog:
- Exceptions not specified
"""

# -------------------------------
# REVERSE QUERY
# -------------------------------

def answer_reverse_query(query):
    courses = extract_courses(query)
    if not courses:
        return None

    target = courses[0]
    result = []

    for course, prereq in prereq_map.items():
        if prereq == target:
            result.append(course)

    if not result:
        return f"""
Answer / Plan:
No courses found requiring {target}

Why:
No matching prerequisite found

Citations:
- None

Clarifying questions:
- None

Assumptions / Not in catalog:
- Data incomplete
"""

    return f"""
Answer / Plan:
Courses requiring {target}: {', '.join(result)}

Why:
These courses list {target} as prerequisite

Citations:{get_citations(result)}

Clarifying questions:
- None

Assumptions / Not in catalog:
- Only direct prerequisites considered
"""

# -------------------------------
# COURSE PLANNING
# -------------------------------

def generate_plan(completed):
    completed_set = expand_completed(completed)

    suggestions = []
    reasons = []

    for course in prereq_map:
        if course in completed_set:
            continue

        chain = get_chain(course)

        if all(pr in completed_set for pr in chain):
            suggestions.append(course)
            reasons.append(
                f"{course}: requires {', '.join(chain) if chain else 'no prerequisites'} (satisfied)"
            )

    return suggestions[:3], reasons[:3]

# -------------------------------
# MAIN LOGIC
# -------------------------------

def answer_question(query):
    _ = retriever.invoke(query)

    courses = extract_courses(query)

    if "after" in query.lower() and courses:
        completed = [courses[0]]
    else:
        completed = courses

    completed = expand_completed(completed)

    # PROGRAM
    if is_program_query(query):
        res = answer_program_question(query)
        if res:
            return res

    # MANDATORY
    if is_mandatory_query(query):
        res = answer_mandatory_question(query)
        if res:
            return res

    # POLICY
    if is_policy_query(query):
        return answer_policy_question(query)

    # REVERSE
    if is_reverse_query(query):
        return answer_reverse_query(query)

    # PLANNING
    if is_planning_query(query):
        plan, reasons = generate_plan(completed)

        return f"""
Answer / Plan:
Suggested courses: {', '.join(plan)}

Why:
{chr(10).join(reasons)}

Citations:{get_citations(plan)}

Clarifying questions:
- Preferred subjects?
- Max credits?

Assumptions / Not in catalog:
- Availability not considered
"""

    # PREREQ / ELIGIBILITY
    if courses:
        target = courses[0]
        chain = get_chain(target)

        if is_prereq_query(query):
            return f"""
Answer / Plan:
{target} requires: {' → '.join(chain) if chain else 'None'}

Why:
Derived from catalog

Citations:{get_citations([target] + chain)}

Clarifying questions:
- None

Assumptions / Not in catalog:
- No grade constraints
"""

        missing = [c for c in chain if c not in completed]

        if not missing:
            return f"""
Answer / Plan:
Eligible

Why:
All prerequisites satisfied: {', '.join(chain)}

Citations:{get_citations([target] + chain)}

Clarifying questions:
- None

Assumptions / Not in catalog:
- Grades not considered
"""

        return f"""
Answer / Plan:
Not eligible

Why:
Missing prerequisites: {', '.join(missing)}

Citations:{get_citations([target] + chain)}

Clarifying questions:
- Have you completed: {', '.join(missing)}?

Assumptions / Not in catalog:
- Grades not considered
"""

    return """
Answer / Plan:
I don't have that information in the provided catalog.

Why:
Not present in documents

Citations:
- None

Clarifying questions:
- Please clarify your query

Assumptions / Not in catalog:
- Data may be incomplete
"""

# -------------------------------
# RUN
# -------------------------------

if __name__ == "__main__":
    query = input("Ask your question: ")
    print(answer_question(query))