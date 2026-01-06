from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage


# Load environment variables
# I use a PDF loader to extract text from the insurance policy document
load_dotenv()

PDF_PATH = "policy.pdf"

# -------------------------
# STEP 1: Load PDF
# -------------------------
print("📄 Loading insurance policy...")
loader = PyPDFLoader(PDF_PATH)
documents = loader.load()

# -------------------------
# STEP 2: Chunking
# -------------------------
# Because policies are very long, I split the text into smaller chunks so the LLM can process them efficiently
print("✂️ Splitting document into chunks...")
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=150
)

chunks = text_splitter.split_documents(documents)

# -------------------------
# STEP 3: Initialize Anthropic LLM
# -------------------------
llm = ChatAnthropic(
    model="claude-3-haiku-20240307",
    temperature=0.3
)

chunk_summaries = []

# -------------------------
# STEP 4: Summarize each chunk
# -------------------------
# Each chunk is sent to the LLM with a prompt asking it to summarize coverage, exclusions, and limits in plain language
for i, chunk in enumerate(chunks):
    print(f"🧠 Summarizing chunk {i+1}/{len(chunks)}")

    prompt = f"""
You are an insurance expert.

From the following insurance policy text, extract key information
related to:
- Coverage
- Exclusions
- Limits

Use simple, plain language.

Policy Text:
{chunk.page_content}
"""

    response = llm.invoke([HumanMessage(content=prompt)])
    chunk_summaries.append(response.content)

# -------------------------
# STEP 5: Combine summaries
# -------------------------
# All chunk summaries are combined and summarized again to generate a final 200-word policy summary.
combined_summary_text = "\n".join(chunk_summaries)

final_prompt = f"""
You are an insurance expert.

Using the combined summaries below, generate a final
plain-language insurance policy summary.

Rules:
- Maximum 200 words
- Avoid legal jargon
- Clearly separate:
  Coverage
  Exclusions
  Limits

Combined Text:
{combined_summary_text}
"""

final_response = llm.invoke([HumanMessage(content=final_prompt)])

# -------------------------
# STEP 6: Output
# -------------------------
print("\n📄 FINAL POLICY SUMMARY\n")
print(final_response.content)
