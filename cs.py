import streamlit as st
from langchain.prompts import PromptTemplate
from langchain_groq import ChatGroq
from langchain.chains.summarize import load_summarize_chain
from langchain_community.document_loaders import PyPDFLoader, YoutubeLoader
from bs4 import BeautifulSoup
from langchain.schema import Document
import validators
import requests
from langchain.text_splitter import RecursiveCharacterTextSplitter
from streamlit_extras.customize_running import center_running
import re

# SVG code
svg_code = '''
<svg fill="#000000" height="100px" width="100px" version="1.1" id="_x31_" xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" viewBox="0 0 303 256" xml:space="preserve">
  <defs>
    <linearGradient id="maroonGoldGradient" x1="0%" y1="0%" x2="100%" y2="100%">
      <stop offset="0%" style="stop-color: maroon; stop-opacity: 1" />
      <stop offset="100%" style="stop-color: gold; stop-opacity: 1" />
    </linearGradient>
  </defs>
  <g id="SVGRepo_bgCarrier" stroke-width="0" />
  <g id="SVGRepo_tracerCarrier" stroke-linecap="round" stroke-linejoin="round" />
  <g id="SVGRepo_iconCarrier">
    <path id="_x33_" fill="url(#maroonGoldGradient)" d="M295.3,130.2c-15.4-4.8-17.2-13.5-22.7-18.8c-11.7-11.7-27.1-1.4-33.2,3.4c-6.6-37.4-33.5-65.1-83.2-65.1 
    c-58.2,0-85.3,38.1-85.3,85.3s28,111.9,85.3,111.9s85.3-64.6,85.3-111.9c0-0.5,0-1.1,0-1.6c10.5,3,11.2,14.9,11.2,14.9 
    s0,24.1,21.3,24.1c-7.8-14.4-2.8-21.8-2.8-29.3c0-4.8-1.6-8.3-3.7-11.2C272.8,135.7,284.7,141.2,295.3,130.2z M221.2,145.5 
    c0,11.7-9.6,21.5-21.3,24.1c-5,1.1-24.5,2.8-44,2.8l0,0l0,0c-19.5,0-39-1.6-44-2.8c-11.7-2.3-21.3-12.2-21.3-24.1v-6 
    c0-6,5-10.5,10.5-10.5c18.6,0,25.4,5.3,54.6,5.3s36-5.3,54.6-5.3c5.5,0,10.5,4.8,10.5,10.5v6H221.2z" />
    <path id="_x32__1_" fill="url(#maroonGoldGradient)" d="M136.9,157.9c-0.5,0-0.7,0-1.4-0.2l-30.3-11c-2.1-0.7-3.2-3-2.3-5c0.7-2.1,3-3.2,5-2.3l30.3,11 
    c2.1,0.7,3.2,3,2.3,5C140.1,157,138.5,157.9,136.9,157.9z" />
    <path id="_x32_" fill="url(#maroonGoldGradient)" d="M175.1,157.9c-1.6,0-3.2-1.1-3.7-2.8c-0.7-2.1,0.2-4.4,2.3-5l30.3-11c2.1-0.7,4.4,0.2,5,2.3 
    c0.7,2.1-0.2,4.4-2.3,5l-30.3,11C176.3,157.9,175.8,157.9,175.1,157.9z" />
    <path id="_x31__1_" fill="url(#maroonGoldGradient)" d="M90.5,48.6c-2.1-2.1-5.3-2.1-7.6,0l-6.6,6.6L29.6,8c-3-3-7.6-3-10.3,0L11,16.2
    c-3,3-3,7.6,0,10.3l46.8,46.8L51.1,80c-2.1,2.1-2.1,5.3,0,7.6c1.1,1.4,2.3,1.8,3.7,1.8s2.8-0.5,3.7-1.6l6.6-6.6l9.4,9.4 
    c0.7-1.6,1.6-3.4,2.3-5l-8-8L80,66.2l6.2,6.2c1.1-1.4,2.3-2.8,3.7-3.9l-6-6l6.6-6.6C92.6,53.8,92.6,50.6,90.5,48.6z 
    M35.8,33.2H25.2V22.7h10.5V33.2z M49.1,46.5H38.5V36h10.5V46.5z M51.6,59.8V49.3h10.5v10.5H51.6z" />
  </g>
</svg>
'''

# Set up the Streamlit app configuration
st.set_page_config(page_title="Summaraii: Cut the Clutter, Keep the Core", page_icon="icon.png", layout="wide")

# CSS for Summaraii styling
st.markdown("""
    <style>
    .main-header {
        color: #B22222;
        text-align: center;
        font-weight: bold;
        font-size: 42px;
    }
    .sub-header {
        text-align: center;
        color: #DAA520;
        font-style: italic;
        margin-bottom: 25px;
        font-size: 18px;
    }
    .stButton>button {
        border-radius: 8px;
        background-color: #B22222;
        color: white;
        font-size: 18px;
        font-weight: bold;
        margin-top: 20px;
    }
    .stTextArea textarea {
        border: 1px solid #FFD700;
        border-radius: 10px;
    }
    </style>
""", unsafe_allow_html=True)

st.markdown("""
    <style>
    .header-container {
        display: flex;
        align-items: center;
        justify-content: center; /* Center the content horizontally */
        gap: 20px;
        margin-bottom: 20px;
    }
    .header-container img {
        height: 100px; /* Adjust the height as needed */
    }
    .main-header {
        font-size: 2.5em;
        color: #B22222; /* Deep Red */
    }
    .sub-header {
        font-size: 1.5em;
        color: #FFD700; /* Gold */
    }
    .description {
        font-size: 1.1em;
        color: #333;
    }
    </style>
""", unsafe_allow_html=True)

st.markdown(f'''
    <div class="header-container">
        <div>{svg_code}</div>
        <div>
            <h1 class="main-header">Summaraii 🗡️</h1>
            <h3 class="sub-header">Cut the Clutter, Keep the Core</h3>
        </div>
    </div>
''', unsafe_allow_html=True)

# Sidebar for Groq API Key and Search Query
with st.sidebar:
    st.markdown(f'<div style="text-align: center;">{svg_code}</div>', unsafe_allow_html=True)
    st.header("  ⚔️ 🗡️ Summaraii 🗡️ ⚔️")
    st.markdown("Summarize reference content for content creators🥷")
    st.subheader("API Configuration")
    groq_api_key = st.text_input("Groq API Key", value="", type="password", placeholder="⚔️Enter Groq API Key here⚔️")
    st.subheader("Topic")
    search_query = st.text_input("Enter the Topic/Title", placeholder="e.g. Python")
    st.subheader("About Summaraii ✒️")
    st.markdown("**Summaraii** delivers a concise summary, focusing only on relevant data, just like a samurai cuts down to the essentials.")
    st.subheader("Tips for Best Results:")
    st.markdown("""
                1. Ensure that the URLs and PDFs you provide contain content relevant to the topic you've specified.
                2. If a URL or PDF does not contain relevant content, Summaraii will notify you.
                3. For optimal summaries, make sure to provide clear and specific topics.
            """)

# Gemma Model Using Groq API
llm = ChatGroq(model="Gemma2-9b-It", groq_api_key=groq_api_key)

# Prompt Template for Summarization
prompt_template = """
Given the topic "{topic}", summarize the key information relevant to the topic from the following content:
Content: {text}
"""
prompt = PromptTemplate(template=prompt_template, input_variables=["text", "topic"])

# Layout for inputs using columns
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("📺 YouTube Video URLs")
    video_urls = st.text_area("Enter YouTube Video URLs (one per line)", placeholder="e.g. https://www.youtube.com/watch?v=dQw4w9WgXcQ" ,height=30)

    st.subheader("🌐 Website URLs")
    website_urls = st.text_area("Enter Website URLs (one per line)", placeholder="e.g. https://www.example.com", height=30)

    st.subheader("📃 Upload PDFs")
    uploaded_files = st.file_uploader("Choose PDF files", type="pdf", accept_multiple_files=True)

with col2:
    st.markdown(
    """
    <h3 style="text-align: center; word-wrap: break-word;">
        ⚔️ Instructions ⚔️
    </h3>
    """, 
    unsafe_allow_html=True
    )
    st.markdown("""
    1. **Topic:** Provide a topic to filter relevant content.
    2. **YouTube Video URLs:** Enter YouTube URLs if any (one per line). 
    3. **Website URLs:** Enter website URLs if any (one per line).
    4. **Upload PDFs:** Upload PDF files if any.
    5. Click the **“Summarize Content”** button to begin the summarization process.
    
    🥷**Summaraii** will process the URLs and PDF files, filter content based on the topic, and generate a concise summary.
    """)

# Helper function to fetch and parse website content
def fetch_website_content(url):
    try:
        response = requests.get(url, headers={"User-Agent": "Mozilla/5.0"})
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')
        paragraphs = soup.find_all('p')
        content = "\n".join([para.get_text() for para in paragraphs])
        return content
    except Exception as e:
        st.error(f"Error fetching website content: {e}")
        return None

# Function to filter content based on search query
def filter_content(content, query):
    if query:
        # Case insensitive search for query in content
        pattern = re.compile(re.escape(query), re.IGNORECASE)
        filtered = "\n".join([line for line in content.split('\n') if pattern.search(line)])
        return filtered
    return content

# Helper function to clean text
def clean_text(text):
    # Remove non-ASCII characters
    text = re.sub(r'[^\x00-\x7F]+', ' ', text)
    return text

# Button to trigger summarization
if st.button("Summarize Content", key="summarize"):
    # Validate Groq API Key
    if not groq_api_key.strip():
        st.error("Please provide the API key.")
        st.stop()
    if not search_query.strip():
        st.error("Please provide topic to search.")
        st.stop()

    combined_documents = []

    # Summarize YouTube Videos
    if video_urls.strip():
        video_urls_list = [url.strip() for url in video_urls.split("\n") if validators.url(url) and "youtube.com" in url]
        for url in video_urls_list:
            try:
                loader = YoutubeLoader.from_youtube_url(url, add_video_info=True)
                docs = loader.load()
                content = "\n".join([doc.page_content for doc in docs])
                filtered_content = filter_content(content, search_query)
                if filtered_content:
                    cleaned_docs = [Document(page_content=clean_text(filtered_content))]
                    combined_documents.extend(cleaned_docs)
                else:
                    st.warning(f"❌ No relevant content found in the Youtube Video {url} for the query: {search_query}")
            except Exception as e:
                st.error(f"Error with {url}: {e}. Maybe try giving a video with Subtitles🤔 ")

    # Summarize Websites
    if website_urls.strip():
        website_urls_list = [url.strip() for url in website_urls.split("\n") if validators.url(url)]
        for url in website_urls_list:
            content = fetch_website_content(url)
            if content:
                filtered_content = filter_content(content, search_query)
                if filtered_content:
                    cleaned_content = clean_text(filtered_content)
                    combined_documents.append(Document(page_content=cleaned_content))
                else:
                    st.warning(f"❌ No relevant content found on {url} for the query: {search_query}")

    # Summarize PDF(s)
    if uploaded_files:
        for uploaded_file in uploaded_files:
            pdf_title = uploaded_file.name
            temp_pdf = f"./temp.pdf"
            with open(temp_pdf, "wb") as file:
                file.write(uploaded_file.getvalue())

            loader = PyPDFLoader(temp_pdf)
            docs = loader.load()

            found_relevant_content = False  # Flag to track if relevant content is found

            for doc in docs:
                filtered_content = filter_content(doc.page_content, search_query)
                if filtered_content:
                    combined_documents.append(Document(page_content=clean_text(filtered_content)))
                    found_relevant_content = True  # Set flag to True when relevant content is found

            if not found_relevant_content:
                st.warning(f"❌ No relevant content found in {pdf_title} for the search query: {search_query}")

    if combined_documents:
        # Split and process the combined documents
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=5000, chunk_overlap=500)
        splits = text_splitter.split_documents(combined_documents)

        # Summarization process using the Groq API
        summaries = []
        chain = load_summarize_chain(llm, chain_type="stuff", prompt=prompt)
        for split in splits:
            doc_text = clean_text(split.page_content)
            summary = chain.run(input_documents=[Document(page_content=doc_text)], topic=search_query)
            summaries.append(summary)

        # Display the summarized content in a read-only editor
        full_summary = "\n\n".join(summaries)
        st.success("✅ Combined Summary of Relevant Sources:")
        st.text_area("🥷 Summary is:", value=full_summary, height=300, disabled=True)
    else:
        st.warning("❌ No valid content found related to the provided topic 😞")
        st.info("Provide content related to topic 🥷 ")