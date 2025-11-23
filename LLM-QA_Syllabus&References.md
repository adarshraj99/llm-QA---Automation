### Fle contains development focused and QA fosed topics. Filter and read: 

#### Topics to learn:
Frameworks: PyTorch(For DL), TenserFlow(For DL), scikit Learn, JAX(For Advanced Researches).
Libraries: Caffe, Microsoft Cognitive tool, D4JS, Numpy, Pandas, matplotlib, Hugging Face.
Tools: Google co-lab provides platform to train ML models and all needed environments.

#### Courses free and HQ todo courses(Courses are listed in order todo for better understanding):
* Google's ML crash course with tensorflow APIs.
* Andrej karphathy zaro to hero series.
* Machine Learning specilization by Andrew Ng and Younes bensounda Mourri.(There is Deep learning Specilization course also by Andrew NG).
* Hugging Face NLP course (for NLP advanced concepts)
* Hands on code prractice (Leetcode) and ML,DL code practice (on Kaggle Quiz and competitions and reddit Quiz Questions)(Check pre-developed projects on kaggle).
* Try implementing needed libraries and it's use cases.
* Try implementing research papers if intrested in research.If, already implemented paper is there, try to implement own way and compare with others. : Or, goto production (implementation) side for maing already deployed apps better if intrested.
* Participate in compititions and looks for other's codes. It will make prepared for v.good interviews also.
* Get good jobs .Do Networking, Keep posting works on linkedIn.

#### Other Free short courses:
* Matrix algera For Engineers(By Jeffery R. Chasnoy)
* Maths: Deeplearning Course for all maths(Intermideate): https://www.coursera.org/specializations/mathematics-for-machine-learning-and-data-science?action=enroll
* Statistics : https://www.coursera.org/learn/stanford-statistics/home/module/1
* EdX Introduction to probablity (By harward University)
* Texas University UTAustinX: Linear Algebra-Foundations to frontiers.
* Khan Academy
* Brillient.org
* projects: Free Samples on Simplilearn youTube Channel: Flower Detection, Face Detection, Parkinsons Disease Detection, Handwriting digit detection.

#### 3Blue1Brown recomendations:
Michael Nielsen book: http://neuralnetworksanddeeplearning.com/ The book walks through the code behind the example in these videos, which you can find here: https://github.com/mnielsen/neural-networks-and-deep-learning
Chris ola blog: https://colah.github.io/
Other topics collections: https://distill.pub/
Other youTube good resources: https://www.youtube.com/watch?v=i8D90DkCLhI and https://www.youtube.com/watch?v=bxe2T-V8XRs
Some ML Algorithms to learn:
naive Bayes.
Support vector Machine
K nearest Neighbour
Linear regression
Logistic regression
Desicion tree
Randonm Forest
K means Clustering


Deepseak nano LLM model link : https://github.com/GeeeekExplorer/nano-vllm/tree/main Deepseak nano LLM model code installation link : pip install git+https://github.com/GeeeekExplorer/nano-vllm.git


#### 📅 Month 1: LLM & GenAI Foundations (Testing Perspective)
Objective: Understand how LLMs work, how to test, what to test, and failure points.

🔹 Concepts
- llm basics (what it is, Tokens vs words and tokenization basics, How transformers work (high-level only: attention, context window), LLM Input → Output pipeline, Deterministic vs non-deterministic systems).
- LLM pipeline: tokenization → model → output → eval
- Prompt engineering basics (few-shot, system/role prompts, Few shot, chain-of-thought, How to structure a prompt, Good vs bad prompt, )
- Hallucination(fabrication of facts), bias, toxicity, relevance, factuality
- QA vs traditional testing: nondeterministic outputs, temperature & max-tokens & top-p (randomness output) and other quality metrics.
- API usage
- Using constraints (JSON output, roles, steps)
- loss of context/ incorrect references.
- Repetition loops
- Tone deviation
- Broken JSON / broken structure
- Overly confident wrong answers
- accuracy, Consistency, Coherence, safety, latency, cost.
- Non-determinism: why the same input → different outputs .Compare 3 prompts for same question
- Evaluating multi-line answers
- Prompt templates


🔹 Tools & Practice
- Use OpenAI GPT-4o or Gemini 1.5 APIs
- Explore LangChain (Python) and LlamaIndex basic workflows
- Start exploring Datasets + Prompts manually (in Jupyter or Python scripts)

🔹 Deliverable
- Notebook comparing 3 prompts on same query, analyzing output accuracy manually.

🔹 Resources
- YouTube: “DeepLearning.AI ChatGPT Prompt Engineering for Devs (Andrew Ng)” (free)
- Book: Deep Learning with Python, 2nd Ed. (for conceptual clarity)
- Blog: “Evaluating LLMs” by OpenAI and Anthropic



#### 📅 Month 2: LLM API Automation + Data Handling
Objective: Automate LLM interactions + store results for validation.

🔹 Skills
- Python API testing (requests, aiohttp)
- JSON parsing, response analysis
- CSV/Excel/JSON datasets as input sources
- Automate prompt → response → store in SQLite/CSV

🔹 Tools
- Python + Pytest
- Pandas (data handling)
- Postman for basic API test
- OpenAI API or HuggingFace Inference API

🔹 Deliverable
- “Prompt Testing Framework v0” — run 50 test prompts automatically and log responses.

🔹 Resources
- OpenAI API docs → “Completions & Evaluations”
- LangChain docs → “LLMChains and OutputParsers”
- YouTube: ArjanCodes / Patrick Loeber (Python projects)



#### 📅 Month 3: Evaluation Frameworks (Core of LLM QA): 
Objective: Learn to score LLM outputs automatically — key product-company need.

🔹 Concepts
- Automatic evaluation: BLEU, ROUGE, cosine similarity
- LLM-as-a-judge techniques (GPT evaluating GPT)
- Human feedback simulation
- Quality metrics: relevance, coherence, factuality, safety

🔹 Tools
- LangSmith (LangChain evaluation platform)
- TruLens, DeepEval, or PromptLayer (for prompt tracking)
- Pandas + Matplotlib for scoring dashboards

🔹 Deliverable
- Build script that scores outputs for relevance using LLM-as-judge + cosine similarity.
- Document results in a mini report (looks good on resume).

🔹 Deliverable
- Build script that scores outputs for relevance using LLM-as-judge + cosine similarity.
- Document results in a mini report (looks good on resume).



#### 📅 Month 4: LLM Automation Pipelines (End-to-End QA Flow)
Objective: Build continuous testing setup similar to product orgs.

🔹 Skills
- Integrate with Pytest or Behave frameworks
- Data-driven testing with LLM outputs
- CI/CD setup (GitHub Actions)
- Mocking & versioning LLM responses

🔹 Tools
- Pytest + Allure reports
- GitHub Actions or Jenkins
- Docker (for reproducibility)
- Vector database (FAISS / Chroma) for context retrieval testing

🔹 Deliverable
- “LLM Automation Framework v1” → Run 100+ tests nightly, auto-report accuracy/failure.

🔹 Resources
- LangChain “Testing & Evaluation” section
- Medium: “Building CI for LLM applications”



📅 Month 5: Domain-Based Mini Projects (Portfolio Building)
Objective: Apply everything in real scenarios.

🔹 Project Ideas
- Chatbot Evaluation Suite – test FAQ bot accuracy, hallucination, tone, response time.
- RAG (Retrieval-Augmented Generation) QA Testing – check if context retrieval works correctly.
- Prompt Regression Testing – ensure updated prompts don’t reduce quality.

🔹 Tools
- OpenAI / HuggingFace models
- LangChain + ChromaDB
- Pytest + TruLens

🔹 Deliverables
- Public GitHub repo with:
- tests/ folder (Pytest automation)
- data/ (prompt sets)
- report.html (auto-eval results)
- YouTube/LinkedIn demo of your project

📅 Month 6: Resume Prep + Mock Interview + Showcase: 
Objective: Polish profile to look like 1 YOE professional.

🔹 Tasks
Prepare 2 project write-ups (Problem, Approach, Framework Diagram, Metrics).
Create short video walkthrough of your framework (YouTube + LinkedIn).
Conduct mock interviews:
LLM pipelines
Prompt engineering
Automation testing patterns
Framework design principles
Demo video link in portfolio.



🧠 EXTRA (Optional but Powerful)
*  Learn Databricks MLflow + model eval tracking (companies love it).
*  Learn Gradio / Streamlit → make a UI for your framework.
*  
