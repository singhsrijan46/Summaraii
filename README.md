# 🥷 Summaraii: Cut the Clutter, Keep the Core

*Summaraii* is a powerful web application that empowers content creators by summarizing relevant information from various reference sources. Whether you're dealing with web URLs, PDFs, or plain text documents, Summaraii leverages advanced Language Learning Models (LLMs) to generate concise summaries tailored to the specific topic or title you provide.

### ⚔️Access Summaraii at: [Summaraii.app](https://summaraii.streamlit.app/)

# ✨ Features

* ⛓️**Multiple Input Types:** Seamlessly handle multiple URLs, PDFs, and text documents.
* 🎯**Topic-Focused Summaries:** Extract and summarize content that directly aligns with your given topic or title.
* ⚡**Efficient Data Extraction:** Eliminate irrelevant information, leaving you with only the key points that matter.
* 👩‍💻**User-Friendly Interface:** Streamlined workflow - input your sources, specify a topic, and receive your summary in a read-only editor.

 # 🚀 Getting Started

_To run the Summaraii application locally, follow these simple steps:_

 # 📋 Prerequisites

**Ensure you have the following installed on your machine:**

* 🐍 Python 3.8 or later
* 📦 Required Python packages (listed in `requirements.txt`)

 # 🔧 Installation

1. **Clone the repository:**

   ```bash
   git clone [invalid URL removed]
   cd summaraii
   ```
2. **Install dependencies:**
    ```bash
   pip install -r requirements.txt
   ```
3. **Set up your environment variables:**
   Create a .env file and provide the necessary API keys and environment variables for LLM and summarization:
   ```bash
   GROQ_API_KEY=<your_groq_api_key>
   ```

 # 🏃‍♂️ Running the Application

 1. **▶️Run the Streamlit app:**
   ```bash
   streamlit run cs.py
   ```
 2. **Access the application:**

After running, navigate to the URL provided by Streamlit (typically http://localhost:8501/) to interact with Summaraii.

# 🛠️ Usage
* 📂 Input Sources: Upload multiple URLs, PDFs, or text documents.
* 📝 Provide Topic/Title: Enter a specific topic or title related to the content.
* ✨ Generate Summary: Summaraii extracts and summarizes relevant data, presenting the information in a read-only editor.
* ⚠️ Error Handling
* ❌ No Relevant Content: If a source doesn’t contain relevant content, the app will issue a warning, and irrelevant data will be excluded from the summary.
* 🤝 Contributing

_Feel free to submit pull requests and suggestions to improve Summaraii. Please ensure all code submissions adhere to the contribution guidelines._

# 📄 License
This project is licensed under the MIT License - see the LICENSE file for details.
