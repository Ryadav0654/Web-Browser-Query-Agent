# === Import Required Libraries ===
import streamlit as st
import os
import asyncio
import google.generativeai as genai

# LangChain & AI-related imports
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_community.vectorstores import FAISS
from langchain.chains.summarize import load_summarize_chain
from langchain.docstore.document import Document

# Web scraping and parsing
from playwright.async_api import async_playwright
from bs4 import BeautifulSoup
from ddgs import DDGS  # DuckDuckGo search wrapper

# === Constants & Configuration ===
VECTOR_STORE_PATH = "faiss_index_google"  # Path to cache FAISS vector store
SIMILARITY_THRESHOLD = 0.8  # Similarity threshold for determining cache hits

# === Google API Configuration ===
try:
    # Load Google API key from Streamlit secrets
    genai.configure(api_key=st.secrets["GOOGLE_API_KEY"])
except Exception as e:
    # Halt execution if key is missing or invalid
    st.error("Google AI API Key not found or invalid. Please check your .streamlit/secrets.toml file.")
    st.stop()

# === Initialize AI Models ===
llm = ChatGoogleGenerativeAI(model="gemini-2.5-pro", temperature=0.7)  # Main language model
embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")  # For semantic similarity

# === Query Validation Function ===
@st.cache_data
def is_query_valid(query: str) -> bool:
    """
    Determines whether a user input is a valid web search query.
    Uses an LLM with a classification prompt.
    """
    prompt_template = """
    You are a query classifier. Your job is to determine if a user's input is a valid search engine query.
    A valid query is something a person would type into Google to find information.
    An invalid query is a command, a greeting, nonsensical text, or a request to perform an action.

    Examples of VALID queries:
    - "Best places to visit in Delhi"
    - "how to bake a chocolate cake"
    - "latest news on AI"

    Examples of INVALID queries:
    - "walk my pet"
    - "add apples to grocery"
    - "hello how are you"
    - "asdfghjkl"

    Now, classify the following query: "{query}"
    Respond with only the word 'VALID' or 'INVALID'.
    """
    validation_prompt = PromptTemplate.from_template(prompt_template)

    # LangChain Expression Language (LCEL) used to streamline chaining
    validation_chain = validation_prompt | llm | StrOutputParser()

    try:
        result_text = validation_chain.invoke({"query": query}).strip().upper()
        st.write(f"Validator Response: {result_text}")  # For debugging
        return "VALID" in result_text
    except Exception as e:
        st.error(f"Error during validation: {e}")
        return False

# === Web Search and Scraping ===
async def search_and_scrape(query: str, num_results: int = 5) -> list[Document]:
    """
    Searches DuckDuckGo for the query and extracts readable content from top URLs.
    Uses Playwright for scraping and BeautifulSoup for parsing.
    """
    documents = []
    st.write(f"🌐 Searching DuckDuckGo for: '{query}'...")

    # Perform search
    with DDGS() as ddgs:
        results = [r for r in ddgs.text(query, max_results=num_results)]

    if not results:
        st.warning("No search results found.")
        return []

    urls = [r['href'] for r in results]
    st.write(f"Found {len(urls)} URLs to scrape.")

    # Use Playwright to scrape each URL asynchronously
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        
        async def scrape_url(url):
            try:
                page = await browser.new_page()
                await page.goto(url, timeout=15000)
                html_content = await page.content()

                soup = BeautifulSoup(html_content, 'html.parser')
                for script_or_style in soup(["script", "style"]):
                    script_or_style.decompose()
                text = soup.get_text()
                lines = (line.strip() for line in text.splitlines())
                chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
                text = "\n".join(chunk for chunk in chunks if chunk)

                if text:
                    return Document(page_content=text, metadata={"source": url})
            except Exception as e:
                st.warning(f"⚠️ Could not scrape {url}: {e}")
                return None
            finally:
                if 'page' in locals() and not page.is_closed():
                    await page.close()

        tasks = [scrape_url(url) for url in urls]
        scraped_docs = await asyncio.gather(*tasks)
        await browser.close()

    # Filter out any None documents
    documents = [doc for doc in scraped_docs if doc]
    st.write(f"✅ Successfully scraped {len(documents)} pages.")
    return documents

# === Document Summarization ===
def summarize_docs(docs: list[Document], query: str) -> str:
    """
    Produces a point-wise, consolidated summary of all documents using Gemini.
    Implements a map-reduce summarization strategy with custom prompts.
    """
    if not docs:
        return "No content available to summarize."

    st.write("🧠 Summarizing content with Google Gemini into a point-wise list...")

    # Define map (extract) and reduce (combine) steps
    map_prompt_template = """
    You are a helpful AI assistant who is an expert at extracting the most important information from a text.
    From the following text, which is related to the query "{query}", please extract the key points.
    
    Focus on facts, figures, and main ideas. Present these key points as a list.
    
    Text:
    "{text}"
    
    KEY POINTS:
    """

    # Define the combine step
    combine_prompt_template = """
You are an expert information synthesizer helping users quickly understand key facts from multiple web sources.

You are given a list of key points extracted from several web pages related to the query: "{query}".

Your task is to create a **final consolidated summary** that is:

1. **Point-wise**, using clear bullet points or numbers.
2. **Concise**: Limit the summary to **100–200 words**.
3. **Factual** and **objective**, avoiding opinions, filler words, or assumptions.
4. **Well-structured**, free from repetition, contradictions, or unnecessary details.
5. **Readable** and **coherent**, suitable for a non-expert reader.
6. Avoid mentioning the sources, URLs, or phrases like “according to a webpage.”

**KEY POINTS (from multiple sources):**
{text}

--- 

**FINAL SUMMARY (in bullet or numbered list):**
"""


    map_prompt = PromptTemplate.from_template(map_prompt_template)
    combine_prompt = PromptTemplate.from_template(combine_prompt_template)

    # Build summarization chain with custom prompts
    chain = load_summarize_chain(
        llm,
        chain_type="map_reduce",
        map_prompt=map_prompt,
        combine_prompt=combine_prompt,
        token_max=4000
    )

    summary = chain.invoke({"input_documents": docs, "query": query})
    return summary['output_text']

# === FAISS Vector Store Handling ===
def load_or_create_vector_store():
    """
    Loads existing FAISS vector index or creates a new one.
    Used for caching summaries based on semantic similarity.
    """
    if os.path.exists(VECTOR_STORE_PATH):
        st.write("Loading existing knowledge base...")
        return FAISS.load_local(VECTOR_STORE_PATH, embeddings, allow_dangerous_deserialization=True)
    else:
        st.write("Creating new knowledge base...")
        dummy_doc = [Document(page_content="start")]
        index = FAISS.from_documents(dummy_doc, embeddings)
        index.delete([index.index_to_docstore_id[0]])  # Remove dummy entry
        return index

# === Streamlit User Interface ===
st.set_page_config(page_title="Ripplica Interview Task (Google AI)", layout="wide")
st.title("🧠 Web Browser Query Agent")
st.markdown("Enter a query, and the agent will search the web, summarize the results, and cache them for future use.")

# Initialize session state
if 'result' not in st.session_state:
    st.session_state.result = None

if 'vector_store' not in st.session_state:
    st.session_state.vector_store = load_or_create_vector_store()

# User input
query = st.text_input("Enter your query here:", key="query_input")

# On Search button click
if st.button("Search", key="search_button"):
    st.session_state.result = None  # Clear previous results

    if not query:
        st.warning("Please enter a query.")
    else:
        final_output = ""
        with st.status("Processing your request...", expanded=True) as status:
            # Step 1: Validate Query
            status.update(label="1/5 - Validating query...")
            if not is_query_valid(query):
                st.error("This is not a valid query. Please enter something you would search on Google.")
                status.update(label="Validation Failed", state="error", expanded=False)
            else:
                # Step 2: Check for cached results in FAISS
                status.update(label="2/5 - Checking cache for similar queries...")
                vector_store = st.session_state.vector_store
                try:
                    similar_docs = vector_store.similarity_search_with_score(query, k=1)

                    if similar_docs and similar_docs[0][1] < SIMILARITY_THRESHOLD:
                        # Use cached summary
                        cached_result = similar_docs[0][0]
                        final_output = f"{cached_result.page_content}"
                        status.update(label="Finished (from cache)!", state="complete", expanded=False)
                    else:
                        # Step 3: Web search & scrape
                        status.update(label="3/5 - Searching the web...")
                        scraped_documents = asyncio.run(search_and_scrape(query))

                        if not scraped_documents:
                            st.error("Could not retrieve any content from the web.")
                            status.update(label="Search Failed", state="error", expanded=False)
                        else:
                            # Step 4: Summarize the scraped content
                            status.update(label="4/5 - Summarizing content...")
                            summary = summarize_docs(scraped_documents, query)
                            final_output = f"{summary}"

                            # Step 5: Cache the result
                            status.update(label="5/5 - Saving result to cache...")
                            new_doc = Document(page_content=summary, metadata={"original_query": query})
                            vector_store.add_documents([new_doc])
                            vector_store.save_local(VECTOR_STORE_PATH)
                            st.session_state.vector_store = vector_store

                            status.update(label="Process Complete!", state="complete", expanded=False)

                except Exception as e:
                    st.error(f"An unexpected error occurred: {e}")
                    status.update(label="Error", state="error", expanded=False)

        st.session_state.result = final_output
        
        
# Display the result outside the button-press logic, if it exists
if st.session_state.result:
    st.markdown(st.session_state.result, unsafe_allow_html=True)
