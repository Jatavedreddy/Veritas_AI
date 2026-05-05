# Product Vision: Veritas Financial Advisory Platform

## 1. Executive Summary

**Veritas** is an enterprise-grade, end-to-end Financial Risk & Advisory Platform designed to modernize how financial institutions manage risk, ensure regulatory compliance, and optimize portfolios. We are moving beyond basic "fraud detection" to build a **Proactive Threat & Advisory Matrix**. By unifying predictive Machine Learning (transaction anomaly detection) with a multi-agent Generative AI architecture (contextual regulatory analysis), Veritas transforms raw financial data into actionable, synthesized intelligence to prevent compliance breaches *before* they settle. The platform is built around a premium, data-journalism-inspired user experience, prioritizing clarity, authority, and rapid decision-making.

## 2. User Personas

* **The Risk Analyst (e.g., Alex)**: Monitors millions of transactions. Needs to rapidly identify anomalous behavior (e.g., trade spoofing, high-velocity money laundering), assess geographic exposure, and investigate flagged clients.
* **The Compliance Officer**: Navigates dense regulatory frameworks (e.g., SEC filings, Basel III, Sanctions lists). Needs instant, accurate answers backed by cited sources to ensure the firm's trading activities do not violate new compliance rules.
* **The Portfolio Manager**: Makes high-stakes investment decisions. Needs a synthesized view of market trends, associated operational risks, and compliance guardrails to confidently allocate assets.

## 3. Core Capabilities & Workflows

### A. ML-Powered Risk & Fraud Engine (The "What")
* **Pipeline**: Automated ingestion of simulated institutional trading streams and retail transactions.
* **Model**: Unsupervised Machine Learning (Isolation Forest / Autoencoders) trained to detect complex anomalies like Trade Spoofing, Wash Trading, and Structuring (Money Laundering). 
* **Output**: Real-time scoring. Highly anomalous transactions are blocked or flagged, appending an "Anomaly Signature" (e.g., 'Volume Spike + Geographic Mismatch').
* **Persistence**: Models serialized (Pickle/ONNX) for real-time inference via a Flask `/predict` REST API.

### B. Multi-Agent AI Advisory System (The "Why" and "What Next")
Built using **CrewAI** and powered by **Groq (Meta Llama 3 models)**, the system features a "Virtual Risk Committee":
1. **The Compliance Auditor Agent**: Equipped with RAG (Azure AI Search). It constantly monitors the knowledge base of SEC regulations and Global Sanctions. When the ML flags a trade, this agent checks if it legally violates a specific statute.
2. **The Quantitative Analyst Agent**: Looks at the portfolio's exposure. If the ML flags systemic anomalies in a specific sector (e.g., Tech stocks in Asia), this agent calculates the potential financial impact on the overall portfolio.
3. **The Chief Risk Officer (CRO) Agent**: The orchestrator. It takes the "Flag" from the ML, the "Legal Context" from the Compliance Agent, and the "Impact" from the Quant Agent, and writes a concise, 3-bullet-point executive brief with a recommendation (e.g., "Freeze Account", "Allow Trade", "Request KYC Update").

### C. Automated SAR (Suspicious Activity Report) & Audit Generator (The "New Feature")
A massive pain point in financial compliance is manual documentation. When a severe anomaly is verified by the Virtual Risk Committee, the GenAI system will automatically draft a legal-grade **Suspicious Activity Report (SAR)**. This covers the transaction timeline, involved entities, and cited regulatory statutes, ready for human review. This saves compliance officers hours of manual paperwork and provides an immutable audit trail for regulators.

### D. Enterprise Data Engineering Pipeline
* **Orchestration**: **Azure Data Factory** pipelines simulating daily batch and micro-batch data ingestion.
* **Storage**: **Azure Blob Storage** (Data Lake architecture).
* **Data Flow**: Medallion Architecture: Bronze (Raw JSON) -> Silver (Cleaned/Filtered) -> Gold (Parquet files aggregated for ML training and BI reporting).
* **Database**: **MongoDB** (via Azure CosmosDB free tier) to store application state, user profiles, agent executive summaries, and active alerts.

### D. Analytics & Visualization (Power BI)
A fully interactive, premium Power BI dashboard embedded in the web app:
* Real-time heatmaps of institutional risk exposure.
* ML Drift monitoring (is the fraud model still accurate?).
* ROI vs. Risk Volatility scatter plots.

## 4. UI/UX Design System
Inspired by high-end financial institutions and premium data journalism (like Bloomberg or WSJ Data).
* **Color Palette**: Off-White (`#FDFAFA`), Obsidian (`#1E1E1E`), Terracotta Red (`#D96B52`), Slate Blue (`#4A6B82`), Muted Sage (`#6B7A6A`).
* **Typography**: Playfair Display (Serif) for authority; Inter/Roboto (Sans-Serif) for dense data readability.
* **Layout**: Minimalist card-based dashboards, prioritizing the "Signal over Noise".

## 5. Technical Architecture & Azure Integration
* **Backend Framework**: Python Flask (APIs: `/ingest`, `/predict`, `/search`, `/advise`).
* **Database**: MongoDB (via Azure CosmosDB).
* **Azure Services Utilized**:
  1. **Azure App Service**: Hosting the Dockerized Flask backend.
  2. **Azure AI Search**: Vector database/indexing for the Compliance Agent's RAG pipeline.
  3. **Azure Data Factory**: Building the ELT data pipelines.
* **Deployment/CI-CD**: GitHub Actions -> Docker -> Azure App Service. Secure management of Groq API keys via Azure Key Vault.

## 6. Implementation Roadmap
* **Phase 1: Foundation & Data Engineering**: Scaffold project, generate synthetic institutional financial data, establish ADF pipeline to Blob Storage.
* **Phase 2: Machine Learning**: EDA, feature engineering, train the Anomaly Detection model for complex fraud, deploy as Flask endpoint.
* **Phase 3: The Brain (GenAI & Agents)**: Set up Groq/CrewAI. Implement the Vector Store (Azure AI Search) with sample SEC filings. Build the 3-agent Virtual Risk Committee.
* **Phase 4: Backend Integration & APIs**: Finalize Flask backend, MongoDB state management, logging, error handling, pytest.
* **Phase 5: The Glass (Frontend & Power BI)**: Develop the premium UI matching the design system. Embed Power BI reports.
* **Phase 6: Cloud Deployment**: Dockerize, GitHub Actions, deploy to Azure App Service, record demo.
