import streamlit as st
from langchain.prompts import PromptTemplate
from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.summarize import load_summarize_chain
from langchain_community.document_loaders import PyPDFLoader, YoutubeLoader
from langchain_community.embeddings import OpenAIEmbeddings, HuggingFaceEmbeddings
from langchain.schema import Document
from bs4 import BeautifulSoup
import validators
import requests
from langchain.text_splitter import RecursiveCharacterTextSplitter
import re
import os
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import json
from typing import Dict, List, Optional, Tuple

# Model Configuration and Pricing
MODEL_CONFIG = {
    "groq": {
        "Gemma2-9b-It": {"cost_per_1k_tokens": 0.0001, "max_tokens": 8192},
        "Llama-3.1-70b": {"cost_per_1k_tokens": 0.0002, "max_tokens": 8192},
        "Mixtral-8x7b": {"cost_per_1k_tokens": 0.0001, "max_tokens": 8192}
    },
    "openai": {
        "gpt-3.5-turbo": {"cost_per_1k_tokens": 0.0015, "max_tokens": 4096},
        "gpt-4": {"cost_per_1k_tokens": 0.03, "max_tokens": 8192},
        "gpt-4-turbo": {"cost_per_1k_tokens": 0.01, "max_tokens": 128000}
    },
    "anthropic": {
        "claude-3-sonnet": {"cost_per_1k_tokens": 0.003, "max_tokens": 200000},
        "claude-3-opus": {"cost_per_1k_tokens": 0.015, "max_tokens": 200000}
    },
    "google": {
        "gemini-pro": {"cost_per_1k_tokens": 0.0005, "max_tokens": 30720}
    }
}

class LLMProvider:
    """Abstract base class for LLM providers"""
    
    def __init__(self, provider_type: str, api_key: str, model_name: str):
        self.provider_type = provider_type
        self.api_key = api_key
        self.model_name = model_name
        self.llm = None
        self._initialize_llm()
    
    def _initialize_llm(self):
        """Initialize the LLM based on provider type"""
        try:
            if self.provider_type == "groq":
                self.llm = ChatGroq(model=self.model_name, groq_api_key=self.api_key)
            elif self.provider_type == "openai":
                self.llm = ChatOpenAI(model=self.model_name, openai_api_key=self.api_key)
            elif self.provider_type == "anthropic":
                self.llm = ChatAnthropic(model=self.model_name, anthropic_api_key=self.api_key)
            elif self.provider_type == "google":
                self.llm = ChatGoogleGenerativeAI(model=self.model_name, google_api_key=self.api_key)
            else:
                raise ValueError(f"Unsupported provider type: {self.provider_type}")
        except Exception as e:
            st.error(f"Failed to initialize {self.provider_type} model {self.model_name}: {e}")
            raise
    
    def summarize(self, content: str, topic: str, options: Dict = None) -> str:
        """Summarize content using the configured model"""
        try:
            prompt_template = """
            Given the topic "{topic}", summarize the key information relevant to the topic from the following content.
            Focus on the most important and relevant points. Be concise but comprehensive.
            
            Content: {text}
            
            Summary:
            """
            prompt = PromptTemplate(template=prompt_template, input_variables=["text", "topic"])
            
            chain = load_summarize_chain(self.llm, chain_type="stuff", prompt=prompt)
            summary = chain.run(input_documents=[Document(page_content=content)], topic=topic)
            return summary
        except Exception as e:
            st.error(f"Error during summarization with {self.provider_type}: {e}")
            raise
    
    def get_cost_estimate(self, content_length: int) -> float:
        """Estimate cost for processing content"""
        if self.provider_type in MODEL_CONFIG and self.model_name in MODEL_CONFIG[self.provider_type]:
            cost_per_1k = MODEL_CONFIG[self.provider_type][self.model_name]["cost_per_1k_tokens"]
            # Rough estimation: 1 token ≈ 4 characters
            estimated_tokens = content_length / 4
            return (estimated_tokens / 1000) * cost_per_1k
        return 0.0

class EnhancedContentProcessor:
    """Enhanced content processing with semantic filtering"""
    
    def __init__(self, filtering_strategy: str = "semantic"):
        self.filtering_strategy = filtering_strategy
        self.vectorizer = None
        self.topic_embedding = None
    
    def filter_content_semantic(self, content: str, topic: str, threshold: float = 0.3) -> str:
        """Filter content using semantic similarity"""
        try:
            # Initialize TF-IDF vectorizer
            if self.vectorizer is None:
                self.vectorizer = TfidfVectorizer(stop_words='english', max_features=1000)
            
            # Split content into sentences
            sentences = re.split(r'[.!?]+', content)
            sentences = [s.strip() for s in sentences if s.strip()]
            
            if len(sentences) < 2:
                return content
            
            # Create TF-IDF matrix
            all_texts = sentences + [topic]
            tfidf_matrix = self.vectorizer.fit_transform(all_texts)
            
            # Calculate similarity between topic and each sentence
            topic_vector = tfidf_matrix[-1]  # Last vector is the topic
            sentence_vectors = tfidf_matrix[:-1]  # All vectors except the last
            
            similarities = cosine_similarity(topic_vector, sentence_vectors)[0]
            
            # Filter sentences based on similarity threshold
            relevant_sentences = []
            for i, similarity in enumerate(similarities):
                if similarity >= threshold:
                    relevant_sentences.append(sentences[i])
            
            return '\n'.join(relevant_sentences) if relevant_sentences else content
            
        except Exception as e:
            st.warning(f"Semantic filtering failed, falling back to regex: {e}")
            return self.filter_content_regex(content, topic)
    
    def filter_content_regex(self, content: str, topic: str) -> str:
        """Original regex-based filtering"""
        if topic:
            pattern = re.compile(re.escape(topic), re.IGNORECASE)
            filtered = "\n".join([line for line in content.split('\n') if pattern.search(line)])
            return filtered
        return content
    
    def filter_content(self, content: str, topic: str, threshold: float = 0.3) -> str:
        """Main filtering method with strategy selection"""
        if self.filtering_strategy == "semantic":
            return self.filter_content_semantic(content, topic, threshold)
        else:
            return self.filter_content_regex(content, topic)
    
    def extract_keywords(self, content: str, num_keywords: int = 10) -> List[str]:
        """Extract key terms from content"""
        try:
            if self.vectorizer is None:
                self.vectorizer = TfidfVectorizer(stop_words='english', max_features=1000)
            
            tfidf_matrix = self.vectorizer.fit_transform([content])
            feature_names = self.vectorizer.get_feature_names_out()
            
            # Get top TF-IDF scores
            scores = tfidf_matrix.toarray()[0]
            top_indices = np.argsort(scores)[-num_keywords:][::-1]
            
            return [feature_names[i] for i in top_indices if scores[i] > 0]
        except Exception as e:
            st.warning(f"Keyword extraction failed: {e}")
            return []

class ModelManager:
    """Manages multiple LLM providers and handles fallbacks"""
    
    def __init__(self):
        self.providers = {}
        self.primary_provider = None
    
    def add_provider(self, provider_type: str, api_key: str, model_name: str, is_primary: bool = False):
        """Add a new provider"""
        try:
            provider = LLMProvider(provider_type, api_key, model_name)
            self.providers[f"{provider_type}_{model_name}"] = provider
            
            if is_primary or self.primary_provider is None:
                self.primary_provider = provider
                
            return True
        except Exception as e:
            st.error(f"Failed to add provider {provider_type}: {e}")
            return False
    
    def summarize_with_fallback(self, content: str, topic: str, options: Dict = None) -> Tuple[str, str]:
        """Summarize with automatic fallback to other providers"""
        providers_to_try = [self.primary_provider] + [p for p in self.providers.values() if p != self.primary_provider]
        
        for provider in providers_to_try:
            if provider is None:
                continue
                
            try:
                summary = provider.summarize(content, topic, options)
                return summary, f"{provider.provider_type}_{provider.model_name}"
            except Exception as e:
                st.warning(f"Provider {provider.provider_type} failed: {e}")
                continue
        
        raise Exception("All providers failed to generate summary")
    
    def get_cost_estimates(self, content_length: int) -> Dict[str, float]:
        """Get cost estimates for all available providers"""
        estimates = {}
        for name, provider in self.providers.items():
            estimates[name] = provider.get_cost_estimate(content_length)
        return estimates

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
        transition: all 0.3s ease;
        border: none;
        padding: 12px 24px;
    }
    .stButton>button:hover {
        background-color: #8B0000;
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(0,0,0,0.2);
    }
    .stButton>button:disabled {
        background-color: #cccccc;
        color: #666666;
        cursor: not-allowed;
        transform: none;
        box-shadow: none;
    }
    .stTextArea textarea {
        border: 2px solid #FFD700;
        border-radius: 10px;
        transition: border-color 0.3s ease;
    }
    .stTextArea textarea:focus {
        border-color: #B22222;
        box-shadow: 0 0 0 2px rgba(178, 34, 34, 0.2);
    }
    .stTextInput>div>div>input {
        border: 2px solid #FFD700;
        border-radius: 8px;
        transition: border-color 0.3s ease;
    }
    .stTextInput>div>div>input:focus {
        border-color: #B22222;
        box-shadow: 0 0 0 2px rgba(178, 34, 34, 0.2);
    }
    .file-uploader {
        border: 2px dashed #FFD700;
        border-radius: 10px;
        padding: 20px;
        text-align: center;
        transition: border-color 0.3s ease;
    }
    .file-uploader:hover {
        border-color: #B22222;
    }
    .status-message {
        padding: 12px;
        border-radius: 8px;
        margin: 10px 0;
        font-weight: 500;
    }
    .loading-spinner {
        display: inline-block;
        width: 20px;
        height: 20px;
        border: 3px solid #f3f3f3;
        border-top: 3px solid #B22222;
        border-radius: 50%;
        animation: spin 1s linear infinite;
        margin-right: 10px;
    }
    @keyframes spin {
        0% { transform: rotate(0deg); }
        100% { transform: rotate(360deg); }
    }
    .progress-container {
        background-color: #f0f0f0;
        border-radius: 10px;
        padding: 20px;
        margin: 20px 0;
    }
    .progress-bar {
        background-color: #B22222;
        height: 20px;
        border-radius: 10px;
        transition: width 0.3s ease;
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
            <h1 class="main-header">Summaraii</h1>
            <h3 class="sub-header">Cut the Clutter, Keep the Core</h3>
        </div>
    </div>
''', unsafe_allow_html=True)

# Sidebar for Multi-Model Configuration and Search Query
with st.sidebar:
    st.markdown(f'<div style="text-align: center;">{svg_code}</div>', unsafe_allow_html=True)
    st.header("Summaraii")
    st.markdown("Summarize reference content for content creators")
    
    # Model Selection
    st.subheader("🤖 Model Configuration")
    model_provider = st.selectbox(
        "Choose Provider",
        ["groq", "openai", "anthropic", "google"],
        help="Select your preferred LLM provider"
    )
    
    # Dynamic model selection based on provider
    if model_provider == "groq":
        model_name = st.selectbox("Groq Model", ["Gemma2-9b-It", "Llama-3.1-70b", "Mixtral-8x7b"])
        api_key = st.text_input("Groq API Key", value="", type="password", placeholder="Enter Groq API Key", help="Get your free API key from https://console.groq.com/")
    elif model_provider == "openai":
        model_name = st.selectbox("OpenAI Model", ["gpt-3.5-turbo", "gpt-4", "gpt-4-turbo"])
        api_key = st.text_input("OpenAI API Key", value="", type="password", placeholder="Enter OpenAI API Key", help="Get your API key from https://platform.openai.com/")
    elif model_provider == "anthropic":
        model_name = st.selectbox("Anthropic Model", ["claude-3-sonnet", "claude-3-opus"])
        api_key = st.text_input("Anthropic API Key", value="", type="password", placeholder="Enter Anthropic API Key", help="Get your API key from https://console.anthropic.com/")
    elif model_provider == "google":
        model_name = st.selectbox("Google Model", ["gemini-pro"])
        api_key = st.text_input("Google API Key", value="", type="password", placeholder="Enter Google API Key", help="Get your API key from https://makersuite.google.com/")
    
    # Advanced Options
    st.subheader("⚙️ Advanced Options")
    filtering_strategy = st.selectbox(
        "Content Filtering",
        ["semantic", "regex"],
        help="Semantic filtering uses AI to find relevant content, regex uses keyword matching"
    )
    
    similarity_threshold = st.slider(
        "Similarity Threshold",
        min_value=0.1,
        max_value=0.9,
        value=0.3,
        step=0.1,
        help="Higher values = more strict filtering (only very relevant content)"
    )
    
    show_cost_estimate = st.checkbox("Show Cost Estimates", value=True, help="Display estimated costs before processing")
    
    # Topic Input
    st.subheader("🎯 Topic")
    search_query = st.text_input("Enter the Topic/Title", placeholder="e.g. Python", help="This will be used to filter relevant content from your sources")
    
    # About Section
    st.subheader("About Summaraii")
    st.markdown("**Summaraii** delivers a concise summary, focusing only on relevant data, just like a samurai cuts down to the essentials.")
    
    st.subheader("Tips for Best Results:")
    st.markdown("""
                1. Ensure that the URLs and PDFs you provide contain content relevant to the topic you've specified.
                2. If a URL or PDF does not contain relevant content, Summaraii will notify you.
                3. For optimal summaries, make sure to provide clear and specific topics.
                4. Try different models to find the best fit for your content type.
            """)

# Legacy prompt template - now handled by LLMProvider class

# Helper function to validate URLs and return validation results
def validate_urls(url_text, url_type="URL"):
    if not url_text.strip():
        return True, [], []  # Valid if empty
    
    urls = [url.strip() for url in url_text.split("\n") if url.strip()]
    valid_urls = []
    invalid_urls = []
    
    for url in urls:
        if url_type == "YouTube" and "youtube.com" not in url:
            invalid_urls.append(f"{url} (not a YouTube URL)")
        elif not validators.url(url):
            invalid_urls.append(f"{url} (invalid URL format)")
        else:
            valid_urls.append(url)
    
    is_valid = len(invalid_urls) == 0
    return is_valid, valid_urls, invalid_urls

# Layout for inputs using columns
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("YouTube Video URLs")
    video_urls = st.text_area("Enter YouTube Video URLs (one per line)", placeholder="e.g. https://www.youtube.com/watch?v=dQw4w9WgXcQ", height=30, help="Enter one YouTube URL per line. Videos must have subtitles for processing.")
    
    # Validate YouTube URLs and show feedback
    video_valid, video_valid_urls, video_invalid_urls = validate_urls(video_urls, "YouTube")
    if video_urls.strip() and not video_valid:
        st.error("Invalid YouTube URLs detected:")
        for invalid_url in video_invalid_urls:
            st.error(f"• {invalid_url}")
    elif video_urls.strip() and video_valid:
        st.success(f"{len(video_valid_urls)} valid YouTube URL(s)")

    st.subheader("Website URLs")
    website_urls = st.text_area("Enter Website URLs (one per line)", placeholder="e.g. https://www.example.com", height=30, help="Enter one website URL per line. Only text content will be extracted.")
    
    # Validate Website URLs and show feedback
    website_valid, website_valid_urls, website_invalid_urls = validate_urls(website_urls, "Website")
    if website_urls.strip() and not website_valid:
        st.error("Invalid Website URLs detected:")
        for invalid_url in website_invalid_urls:
            st.error(f"• {invalid_url}")
    elif website_urls.strip() and website_valid:
        st.success(f"{len(website_valid_urls)} valid Website URL(s)")

    st.subheader("Upload PDFs")
    uploaded_files = st.file_uploader("Choose PDF files", type="pdf", accept_multiple_files=True, help="Upload one or more PDF files. Text will be extracted and filtered by your topic.")

with col2:
    st.markdown(
    """
    <h3 style="text-align: center; word-wrap: break-word;">
        Instructions
    </h3>
    """, 
    unsafe_allow_html=True
    )
    st.markdown("""
    1. **Topic:** Provide a topic to filter relevant content.
    2. **YouTube Video URLs:** Enter YouTube URLs if any (one per line). 
    3. **Website URLs:** Enter website URLs if any (one per line).
    4. **Upload PDFs:** Upload PDF files if any.
    5. Click the **"Summarize Content"** button to begin the summarization process.
    
    **Summaraii** will process the URLs and PDF files, filter content based on the topic, and generate a concise summary.
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

# Legacy function - now handled by EnhancedContentProcessor

# Helper function to clean text
def clean_text(text):
    # Remove non-ASCII characters
    text = re.sub(r'[^\x00-\x7F]+', ' ', text)
    return text

# Check if all inputs are valid
all_urls_valid = video_valid and website_valid
has_content = (video_urls.strip() or website_urls.strip() or uploaded_files)
can_summarize = all_urls_valid and has_content and api_key.strip() and search_query.strip()

# Show helpful message when button is disabled
if not can_summarize:
    if not api_key.strip():
        st.warning("Please provide your API key to continue")
    elif not search_query.strip():
        st.warning("Please provide a topic to search for")
    elif not all_urls_valid:
        st.warning("Please fix invalid URLs before summarizing")
    elif not has_content:
        st.info("Please provide at least one URL or upload a PDF to summarize")

# Cost estimation display
if show_cost_estimate and has_content and api_key.strip():
    st.subheader("💰 Cost Estimation")
    
    # Calculate total content length for estimation
    total_content_length = 0
    if video_urls.strip() and video_valid:
        total_content_length += len(video_urls) * 1000  # Rough estimate
    if website_urls.strip() and website_valid:
        total_content_length += len(website_urls) * 500  # Rough estimate
    if uploaded_files:
        total_content_length += sum(len(f.getvalue()) for f in uploaded_files)
    
    if total_content_length > 0:
        # Create a temporary provider for cost estimation
        try:
            temp_provider = LLMProvider(model_provider, api_key, model_name)
            estimated_cost = temp_provider.get_cost_estimate(total_content_length)
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Estimated Cost", f"${estimated_cost:.4f}")
            with col2:
                st.metric("Content Length", f"{total_content_length:,} chars")
            with col3:
                st.metric("Model", f"{model_provider}/{model_name}")
        except Exception as e:
            st.warning(f"Could not estimate cost: {e}")

# Button to trigger summarization
if st.button("Summarize Content", key="summarize", disabled=not can_summarize):
    # Validate API Key
    if not api_key.strip():
        st.error("Please provide the API key.")
        st.stop()
    if not search_query.strip():
        st.error("Please provide topic to search.")
        st.stop()

    # Initialize enhanced content processor
    content_processor = EnhancedContentProcessor(filtering_strategy=filtering_strategy)
    
    # Initialize model manager and provider
    model_manager = ModelManager()
    success = model_manager.add_provider(model_provider, api_key, model_name, is_primary=True)
    
    if not success:
        st.error(f"Failed to initialize {model_provider} model {model_name}")
        st.stop()
    
    # Initialize progress tracking
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    combined_documents = []
    total_sources = 0
    processed_sources = 0
    
    # Count total sources to process
    if video_urls.strip() and video_valid:
        total_sources += len(video_valid_urls)
    if website_urls.strip() and website_valid:
        total_sources += len(website_valid_urls)
    if uploaded_files:
        total_sources += len(uploaded_files)
    
    if total_sources == 0:
        st.warning("No valid sources to process.")
        st.stop()

    # Summarize YouTube Videos
    if video_urls.strip() and video_valid:
        for i, url in enumerate(video_valid_urls):
            status_text.text(f"Processing YouTube video {i+1}/{len(video_valid_urls)}: {url[:50]}...")
            try:
                loader = YoutubeLoader.from_youtube_url(url, add_video_info=True)
                docs = loader.load()
                content = "\n".join([doc.page_content for doc in docs])
                filtered_content = content_processor.filter_content(content, search_query, similarity_threshold)
                if filtered_content:
                    cleaned_docs = [Document(page_content=clean_text(filtered_content))]
                    combined_documents.extend(cleaned_docs)
                    
                    # Extract keywords for better context
                    keywords = content_processor.extract_keywords(filtered_content, 5)
                    if keywords:
                        st.info(f"Key terms found in {url[:30]}...: {', '.join(keywords)}")
                else:
                    st.warning(f"No relevant content found in the Youtube Video {url} for the query: {search_query}")
            except Exception as e:
                st.error(f"Error with {url}: {e}. Maybe try giving a video with Subtitles")
            
            processed_sources += 1
            progress_bar.progress(processed_sources / total_sources)

    # Summarize Websites
    if website_urls.strip() and website_valid:
        for i, url in enumerate(website_valid_urls):
            status_text.text(f"Processing website {i+1}/{len(website_valid_urls)}: {url[:50]}...")
            content = fetch_website_content(url)
            if content:
                filtered_content = content_processor.filter_content(content, search_query, similarity_threshold)
                if filtered_content:
                    cleaned_content = clean_text(filtered_content)
                    combined_documents.append(Document(page_content=cleaned_content))
                    
                    # Extract keywords for better context
                    keywords = content_processor.extract_keywords(filtered_content, 5)
                    if keywords:
                        st.info(f"Key terms found in {url[:30]}...: {', '.join(keywords)}")
                else:
                    st.warning(f"No relevant content found on {url} for the query: {search_query}")
            
            processed_sources += 1
            progress_bar.progress(processed_sources / total_sources)

    # Summarize PDF(s)
    if uploaded_files:
        for i, uploaded_file in enumerate(uploaded_files):
            pdf_title = uploaded_file.name
            status_text.text(f"Processing PDF {i+1}/{len(uploaded_files)}: {pdf_title[:30]}...")
            
            temp_pdf = f"./temp.pdf"
            with open(temp_pdf, "wb") as file:
                file.write(uploaded_file.getvalue())

            loader = PyPDFLoader(temp_pdf)
            docs = loader.load()

            found_relevant_content = False  # Flag to track if relevant content is found

            for doc in docs:
                filtered_content = content_processor.filter_content(doc.page_content, search_query, similarity_threshold)
                if filtered_content:
                    combined_documents.append(Document(page_content=clean_text(filtered_content)))
                    found_relevant_content = True  # Set flag to True when relevant content is found

            if not found_relevant_content:
                st.warning(f"No relevant content found in {pdf_title} for the search query: {search_query}")
            else:
                # Extract keywords for better context
                pdf_content = "\n".join([doc.page_content for doc in docs])
                keywords = content_processor.extract_keywords(pdf_content, 5)
                if keywords:
                    st.info(f"Key terms found in {pdf_title[:30]}...: {', '.join(keywords)}")
            
            # Clean up temporary file
            if os.path.exists(temp_pdf):
                os.remove(temp_pdf)
            
            processed_sources += 1
            progress_bar.progress(processed_sources / total_sources)

    if combined_documents:
        # Update status for summarization phase
        status_text.text("Generating summary from processed content...")
        progress_bar.progress(0.9)
        
        # Split and process the combined documents
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=5000, chunk_overlap=500)
        splits = text_splitter.split_documents(combined_documents)

        # Summarization process using the selected model with fallback
        summaries = []
        used_model = None
        
        for i, split in enumerate(splits):
            status_text.text(f"Summarizing chunk {i+1}/{len(splits)}...")
            doc_text = clean_text(split.page_content)
            
            try:
                summary, model_used = model_manager.summarize_with_fallback(doc_text, search_query)
                summaries.append(summary)
                if used_model is None:
                    used_model = model_used
            except Exception as e:
                st.error(f"Failed to summarize chunk {i+1}: {e}")
                summaries.append(f"[Error summarizing chunk {i+1}]")

        # Complete progress
        progress_bar.progress(1.0)
        status_text.text("Summary generation complete!")
        
        # Display the summarized content in a read-only editor
        full_summary = "\n\n".join(summaries)
        
        # Clear progress indicators
        progress_bar.empty()
        status_text.empty()
        
        # Show source statistics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Sources Processed", total_sources)
        with col2:
            st.metric("Relevant Sources Found", len(combined_documents))
        with col3:
            st.metric("Summary Chunks", len(splits))
        with col4:
            st.metric("Model Used", used_model or "Unknown")
        
        st.success("Combined Summary of Relevant Sources:")
        
        # Create expandable sections for better organization
        with st.expander("View Summary", expanded=True):
            st.text_area("Summary:", value=full_summary, height=300, disabled=True, key="summary_display")
        
        # Add download option
        col1, col2 = st.columns([1, 1])
        with col1:
            st.download_button(
                label="📄 Download Summary as Text",
                data=full_summary,
                file_name=f"summaraii_summary_{search_query.replace(' ', '_')}.txt",
                mime="text/plain",
                use_container_width=True
            )
        with col2:
            if st.button("🔄 Generate New Summary", use_container_width=True):
                st.rerun()
    else:
        st.warning("No valid content found related to the provided topic")
        st.info("Provide content related to topic")