cosmosProject instructions given by trainer to me (student) :

---

### 1. Project Overview

Students will design, build, and deploy an **end-to-end Multi-Agent AI Platform** that integrates:

- Python Full-stack development
- Machine Learning / Deep Learning
- GenAI (LLMs, RAG, embeddings, multi-agent orchestration)
- Azure AI services
- Azure Data Engineering & Analytics stack
- Power BI visualization
- Azure deployment / CI-CD fundamentals

The project simulates a real-world Data + AI engineering scenario, requiring the student to integrate skills learned across the full track.

### 2. Problem Statement (Choose ONE Domain)

Students must pick one domain and build their end-to-end solution around it:

1. **Smart Retail Assistant** – Demand forecasting + customer Q&A + anomaly detection
2. **Intelligent Healthcare Support System** – Symptom triage + medical information assistant + patient data analytics
3. **Financial Risk & Advisory Platform** – Fraud detection + portfolio insights + regulatory search assistant
4. **Manufacturing Quality & Productivity Suite** – Defect classification + production optimization + maintenance insights
5. **Custom domain scenario** (with SME consent)

### 3. Technical Requirements

### A. Python Fullstack (Backend + APIs)

Students must create:

- Python-based backend using **Flask or FastAPI**(i choose flask)
- Minimum **4 REST APIs** (data ingestion, ML prediction, document search, agent interaction)
- Integration with any one database:
  - **SQL** (Azure SQL) or
  - **NoSQL** (MongoDB, CosmosDB) (i prefer MongoDB)
- Basic **logging, error handling, and unit testing (pytest)**

### B. Machine Learning / Deep Learning

Implement at least **one ML/DL model**, e.g.:

- Regression / classification (sklearn)
- Clustering
- Time-series forecasting
- Image classification (CNN / OpenCV)

**Deliverables:**

- Clean data pipeline
- Feature engineering
- Training + evaluation
- Model persistence (pickle / ONNX)

### C. GenAI / Agents

Build a **minimum 2–3 agent system** using:

- Prompt engineering
- Embeddings + vector store
- Retrieval-Augmented Generation (RAG)
- Agents (Autogen / LangChain / CrewAI / Azure GenAI)

**Examples:**

- **Data Analyst Agent** – answers questions from analytics data
- **Document Assistant Agent** – searches PDFs / knowledge base
- **ML Expert Agent** – helps generate insights from ML outputs

**Implement:**

- Multi-agent orchestration

### D. Azure AI & Cloud

Use at least **three Azure components**:

- **Azure OpenAI / Azure AI Studio**
- **Azure Bot Service** or Web App
- **Azure Cognitive Services** (Search, Vision, Document Intelligence, Text Analytics)
- **Azure ML** for training or inference
- **Azure AI Foundry** for deployment patterns

**Include:**

- Deployment diagram
- Security consideration (Key Vault, environment variables)

### E. Data Engineering Pipeline

Using any combination of:

- **Azure Data Factory** – ingest raw data
- **Azure Databricks (PySpark)** – clean, transform, analyze data
- **Azure Fabric** – Lakehouse, pipelines, Data Activator
- **SQL** (Spark SQL or T-SQL)

**Deliver:**

- Raw → Staged → Curated data flow
- Delta tables or parquet-based storage

### F. Analytics & Visualization

Build a **Power BI dashboard** showing:

- Key metrics
- Model outputs
- Anomaly alerts / trends
- Agent-driven insights (optional)
- Publish & share report

### G. Final Deployment

Deploy the solution on Azure using one of:

- **Azure Web App / Container App**
- **Docker + GitHub Actions CI/CD**
- **Azure App Service**
- **Serverless Functions** (for tasks)

**Your multi-agent system must be live / executable during evaluation.**

### 4. Deliverables

| Deliverable                       | Description                                            |
| --------------------------------- | ------------------------------------------------------ |
| **Technical Documentation** | Architecture, data flow, models, prompts, agents, APIs |
| **Working Code Repository** | GitHub link with clear folder structure                |
| **Deployment Diagram**      | Azure components + interactions                        |
| **Configuration Files**     | Environment, pipeline definitions, YAML, etc.          |
| **Power BI Report**         | Final analytics dashboard                              |
| **Demo Video (5–10 mins)** | Walkthrough of platform                                |
| **Reflection Note**         | Challenges + learnings + optimizations                 |

Instructions for you(LLM/copilot) :
I will be conversing with you to generate prompts to give to github copilot or similar ai tool , to build the project . I will ask you a prompt , and acknowledge the built result . And then we move forward to next step . Lets first setup the idea , design / architecure of the project before we move on to that
We will do it properly the way great developrs do it .
