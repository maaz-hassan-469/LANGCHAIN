# 🤖 LangChain Ecosystem: Generative AI, RAG & Autonomous Agents

Welcome to my comprehensive LangChain development repository! 🚀 This space tracks my architectural journey of mastering Large Language Models (LLMs), LangChain Expression Language (LCEL), Vector Embeddings, Semantic Search, and structural data components to engineer production-ready GenAI Chatbots, Custom AI Agents, and robust Retrieval-Augmented Generation (RAG) applications.

---

## 📂 Repository Layout & Module Directory

Below is the complete mapping of my current architectural modules. Each section contains a dedicated table listing my script files along with a description of their technical role in the application.

### 🔗 Chains
*Exploring classic LangChain components and chain orchestration configurations.*

| Files Contained | Description / What's Happening |
| :--- | :--- |
| `conditional_chain.py` | Routes the execution path dynamically to different sub-chains based on intermediate conditions or analysis. |
| `parallel_chain.py` | Executes multiple operations or multiple prompt invocations simultaneously to optimize processing speed. |
| `sequential_chain.py` | Chains components together step-by-step, passing the exact output of one chain as the input to the next. |
| `simple_chain.py` | Core building block demonstrating a straightforward Prompt + LLM user interaction. |

---

### 💬 Chatmodels
*Integrating Chat-based API configurations across major foundational enterprise models.*

| Files Contained | Description / What's Happening |
| :--- | :--- |
| `1.chatmodel_openai.py` | Instantiates and handles conversational parameters utilizing the OpenAI GPT models. |
| `2.chatmodel_anthropic.py` | Integrates with Anthropic's Claude API to configure advanced long-context chat responses. |
| `3.chatmodel_google.py` | Handles standard setup and prompt messaging logic for Google's Gemini models. |
| `4.chatmodel_huggingf_api.py` | Connects remotely via serverless endpoints to open-source foundation models hosted on Hugging Face Hub. |
| `5.chatmodel_huggingf_local.py` | Downloads, caches, and runs open-source LLMs locally on hardware using Hugging Face pipelines. |

---

### 📂 Document Loader
*Ingesting multi-source structural and unstructured enterprise data formats for downstream context.*

| Files Contained | Description / What's Happening |
| :--- | :--- |
| `directory_loader.py` | Scans an entire local folder path to bulk-ingest all files matching specific file types at once. |
| `lazy_load.py` | Loads files one chunk at a time sequentially to preserve physical system RAM when parsing massive logs. |
| `PDF.pdf` | Sample raw document binary utilized to verify and benchmark local parser logic. |
| `pypdfloader.py` | Explicit PDF integration using PyPDF to split documents page-by-page into structured arrays. |
| `text_loader.py` | Direct pipeline to extract raw strings from standard, unformatted text files into LangChain Document objects. |
| `text.txt` | Core test text playground configuration used for sanity-checking load operations. |
| `webbase_loader.py` | Scrapes raw HTML structures directly from URLs, stripping headers/scripts to isolate textual core content. |

---

### 🧬 EmbeddingModels
*Generating semantic mathematical representations and cross-document mathematical similarities.*

| Files Contained | Description / What's Happening |
| :--- | :--- |
| `1.embedding_query_openai.py` | Generates a 1D vector matrix representation for single string user questions using OpenAI API embeddings. |
| `2.embedding_docs_openai.py` | Generates a batch collection of numerical vectors for multi-page documents to index into a vector store. |
| `3.embedding_hf_local.py` | Runs light-weight embedding layers locally (e.g., SentenceTransformers) without requiring external web requests. |
| `4.document_similarity.py` | Computes cosine similarity or dot product metrics between vectors to retrieve the most contextually relevant documents. |

---

### 🤖 LLMs
*Working with legacy/base text-completion level Large Language Models.*

| Files Contained | Description / What's Happening |
| :--- | :--- |
| `1.LLM.py` | Implements traditional completion models (plain text in, plain text out) instead of modern chat structures. |

---

### 🛠️ Output Parsers
*Converting raw unstructured system string responses into strictly formatted objects.*

| Files Contained | Description / What's Happening |
| :--- | :--- |
| `json_outputparser.py` | Intercepts system text output streams and strictly formats them into valid, nestable JSON key-value blocks. |
| `pydantic_outputparser.py` | Synchronizes LLM text output strings with a predefined typed schema, validating parameters automatically. |
| `str_outparser2.py` | Alternative text parsing workflow variant constructed to filter additional diagnostic chat logs. |
| `str_outputparser.py` | Standard parser layer that extracts only the response text string, ignoring metadata envelopes. |
| `structure_outputparser.py` | Older structural schema utility providing structured mapping over custom key arrays. |

---

### ⚡ Primitive Runnables
*Deep dive into LangChain Expression Language (LCEL) runtime engine primitives.*

| Files Contained | Description / What's Happening |
| :--- | :--- |
| `runnable_branch.py` | Implements routing conditions directly within a pipeline utilizing `RunnableBranch` definitions. |
| `runnable_lambda.py` | Wraps custom, standard Python functions seamlessly into LCEL compatible processing pipes using `RunnableLambda`. |
| `runnable_parallel.py` | Maps independent processing threads dynamically in parallel via an explicit JSON dictionary layout. |
| `runnable_passthrough.py` | Passes original system inputs safely down the line unchanged while generating concurrent computational data. |
| `sequence_runnables.py` | Demonstrates the core behavior of piping operators `\|` to chain multiple runnables back-to-back. |

---

### 📝 Prompts
*Assembling systematic runtime context, chat management templates, and chat states.*

| Files Contained | Description / What's Happening |
| :--- | :--- |
| `chat_bot.py` | Complete orchestration pipeline containing working, iterative conversation logic loops. |
| `chat_history.txt` | Flat file database tracking system history lines to preserve chat history during terminal sessions. |
| `chat_prompt_template.py` | Configures structured message templates (`SystemMessage`, `HumanMessage`) with dynamic slot replacements. |
| `message_placeholder.py` | Inserts variable-length message lists or complete thread states into prompt setups dynamically. |
| `messages.py` | Baseline manual construction script analyzing conversational data block structural arrays. |

---

### 🏗️ Structured Output
*Enforcing strict JSON schemas and validation contracts using native `.with_structured_output()` syntax.*

| Files Contained | Description / What's Happening |
| :--- | :--- |
| `pydantic_demo.py` | Base script practicing schema blueprints using Pydantic BaseModel logic objects. |
| `typeddict_demo.py` | Utilizes Python type-hint definitions (`TypedDict`) to enforce strict key signatures. |
| `withstructuredoutput_typeddict.py` | Forces the LLM instance to respond exclusively within a structure schema bounded by a TypedDict target. |
| `withstructuredoutput_json.py` | Leverages model JSON-mode features to isolate safe configuration schemas from raw strings. |
| `withstructuredoutput_pydantic.py` | The most robust structured setup; binds a class blueprint to guarantee the AI response maps perfectly to runtime objects. |

---

### ✂️ Text Splitter
*Chunking massive text files into manageable overlapping token segments for downstream embedding injections.*

| Files Contained | Description / What's Happening |
| :--- | :--- |
| `lengthbasetext_splitter.py` | E.g., Using `RecursiveCharacterTextSplitter` to separate texts into chunk sizes of 500 with a 50-token overlap. |

---

## ⚙️ Initial Configuration & Setup

1. **Clone the repository ecosystem:**
   ```bash
   git clone https://github.com/maaz-hassan-469/Local-RAG-Pipeline
Install dependency packages:

Bash
pip install -r requirements.txt
Environment Setup:
Create an .env file inside your root directory containing your developer access keys:

Code snippet
OPENAI_API_KEY="your_openai_key"
ANTHROPIC_API_KEY="your_anthropic_key"
GOOGLE_API_KEY="your_google_gemini_key"
HUGGINGFACEHUB_API_TOKEN="your_huggingface_token"
📈 Connect with Me
Let's discuss conversational interfaces, agentic structures, and vector indexes:

Email: maazhassaan469@gmail.com

LinkedIn: www.linkedin.com/in/maaz-hassan-690a97359
