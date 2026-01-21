import streamlit as st
from openai import OpenAI
from groq import Groq

st.set_page_config(page_title = "Hallucination Detector", layout='wide')
st.title("LLM Hallucination Detector")

st.sidebar.header("API Credentials")

openai_key = st.sidebar.text_input(
    "OPENAI API KEY",
    type="password"
)
groq_key = st.sidebar.text_input(
    "GROQ API KEY",
    type="password"
)

#missing = []
if not openai_key:
    st.warning("Missing OpenAI API KEY")
    st.stop()
if not groq_key:
    st.warning("Missing Groq API KEY")
    st.stop()

openai = OpenAI(api_key=openai_key)
groqai = Groq(api_key=groq_key)

# st.subheader("Ask Question")

# question = st.text_area(
#     "Enter you question:",
#     height=100,
# )

# if not question.strip():
#     st.info("Enter a question to start.")
#     st.stop()

# st.subheader("Generated Answer")
def generate_answer(question: str) -> str:
    response = groqai.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[
            {"role": "system", "content": "You are a helpful AI assistant."},
            {"role":"user", "content": question}
        ],
        temperature = 0.7
    )
    return response.choices[0].message.content


def extract_claims(answer: str) -> list[str]:
    messages = [
        {
            "role": "system",
            "content": (
                "Extract all the factual claims from the answer. \n"
                "Return each claim as a bullet point. \n"
                "Do NOT add explanations."
            )
        },
        {
            "role": "user",
            "content": answer
        }
    ]

    response = openai.chat.completions.create(
        model = "gpt-4o-mini",
        messages = messages,
        temperature=0
    )

    claims_text = response.choices[0].message.content

    claims = [
        c.strip("- ").strip()
        for c in claims_text.split("\n")
        if c.strip()
    ]

    return claims

def verify_claim(claim: str) -> str:

    messages = [
        {
            "role": "system",
            "content": (
                "Verify the factual accuracy of the claim. \n"
                "Reply with ONLY ONE WORD: TRUE or FALSE."
            )
        },
        {
            "role": "user",
            "content": f"Claim: {claim}"
        }
    ]

    response = openai.chat.completions.create(
        model = "gpt-4o-mini",
        messages=messages,
        temperature=0
    )

    # verdict = response.choices[0].message.content.strip().upper()
    # if verdict not in ["TRUE", "FALSE"]:
    #     verdict = "FALSE"
    # return verdict
    raw_verdict = response.choices[0].message.content.upper()

    if "TRUE" in raw_verdict:
        return "TRUE"
    elif "FALSE" in raw_verdict:
        return "FALSE"
    else:
        return "FALSE"


def verify_claims(claims: list[str]) -> list[tuple[str, str]]:
    results = []
    for claim in claims:
        verdict = verify_claim(claim)
        results.append((claim, verdict))
    return results

#def agent_decision(verification_results: list[tuple[str, str]]) -> bool:

#    return any(verdict== "FALSE" for _, verdict in verification_results)
    #return any(str(verdict).strip().upper() != "TRUE" for _, verdict in verification_results)
def agent_decision(verification_results: list[tuple[str, str]]) -> bool:
    return any(verdict == "FALSE" for _, verdict in verification_results)

st.subheader("Ask a Question")

question = st.text_area(
    "Enter your question:",
    height=100,
    placeholder= "Who invented the telephone?"
)

if not question.strip():
    st.info("Enter a question to start.")
    st.stop()

st.subheader(" Generated Answer")
answer = generate_answer(question)
st.write(answer)

st.subheader("Extracted Claims")
claims = extract_claims(answer)

if not claims:
    st.warning("No factual claims detected.")
    st.stop()

for c in claims:
    st.write(f"- {c}")

st.subheader("Claim Verification")
verification_results = verify_claims(claims)

for claim, verdict in verification_results:
    if verdict == "TRUE":
        st.success(f"Correct- {claim}")
    else:
        st.error(f"Wrong- {claim}")

st.subheader("Final Verdict")
hallucinated = agent_decision(verification_results)

if hallucinated:
    st.error("Hallucination Detected")
else:
    st.success("Answer appears factual")