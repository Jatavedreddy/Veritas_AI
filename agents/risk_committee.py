"""
Veritas — Virtual Risk Committee (Multi-Agent System)
══════════════════════════════════════════════════════
Three-agent sequential crew that processes ML-flagged transactions:

  Agent 1: Compliance Auditor    → RAG-powered regulatory lookup
  Agent 2: Quantitative Analyst  → Financial impact assessment
  Agent 3: Chief Risk Officer    → Executive brief + SAR draft

Powered by:
  - CrewAI for agent orchestration
  - Groq (Llama 3) for fast LLM inference
  - Azure AI Search + sentence-transformers for RAG

Usage:
    python -m agents.risk_committee                       # sample flagged transaction
    python -m agents.risk_committee --transaction '{}'    # custom JSON input
"""

import os
import sys
import json
import argparse
from typing import Type, Optional

from dotenv import load_dotenv

load_dotenv()

# ─── Configuration ─────────────────────────────────────────────────────────────

GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GROQ_MODEL = os.getenv("GROQ_MODEL", "groq/compound")
AZURE_SEARCH_ENDPOINT = os.getenv("AZURE_SEARCH_ENDPOINT", "")
AZURE_SEARCH_KEY = os.getenv("AZURE_SEARCH_KEY", "")
AZURE_OPENAI_EMBEDDING_DEPLOYMENT = os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT", "text-embedding-3-small")
INDEX_NAME = "veritas-regulations"

# Local fallback path for regulatory embeddings (when Azure is unavailable)
LOCAL_EMBEDDINGS_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "..", "data", "processed", "regulatory_embeddings.json",
)


# ─── Utilities ─────────────────────────────────────────────────────────────────

def _header(title: str) -> None:
    width = 60
    print(f"\n{'─' * width}")
    print(f"  {title}")
    print(f"{'─' * width}")


def _step(msg: str) -> None:
    print(f"  ▸ {msg}")


# ─── Sample Flagged Transaction (from Phase 2 ML model) ──────────────────────

SAMPLE_FLAGGED_TRANSACTION = {
    "transaction_id": "f47ac10b-58cc-4372-a567-0e02b2c3d479",
    "timestamp": "2026-03-15T02:47:00Z",
    "from_account": "INST-4808-3136-7837",
    "to_account": "UNKN-9999-0001-7722",
    "amount": 9998.50,
    "currency": "USD",
    "transaction_type": "ACH",
    "region": "North America",
    "anomaly_score": -0.823,
    "anomaly_type": "structuring",
    "ml_confidence": 0.94,
    "related_transactions": [
        {"amount": 9995.00, "time_delta_minutes": 12},
        {"amount": 9999.00, "time_delta_minutes": 37},
        {"amount": 9992.00, "time_delta_minutes": 65},
        {"amount": 9997.50, "time_delta_minutes": 91},
    ],
    "account_metadata": {
        "account_holder": "Global Trade Solutions LLC",
        "account_age_days": 45,
        "historical_avg_transaction": 2340.00,
        "country_of_registration": "United States",
    },
}


# ─── Regulatory Search Tool (RAG) ────────────────────────────────────────────

def _build_regulatory_search_tool():
    """
    Build the RegulatorySearchTool for the Compliance Auditor Agent.
    Supports two modes:
      1. Azure AI Search (production) — vector search against the cloud index
      2. Local fallback (development)  — cosine similarity against saved embeddings
    """

    from pydantic import BaseModel, Field
    from crewai.tools import BaseTool

    class RegulatorySearchInput(BaseModel):
        """Input schema for the RegulatorySearchTool."""
        query: str = Field(
            ...,
            description=(
                "A natural-language search query about financial regulations, "
                "compliance rules, or legal statutes. For example: "
                "'What are the structuring laws under the Bank Secrecy Act?'"
            ),
        )

    class RegulatorySearchTool(BaseTool):
        name: str = "Regulatory Search Tool"
        description: str = (
            "Searches a curated knowledge base of financial regulations including "
            "Basel III, SEC Rule 10b-5, Bank Secrecy Act, OFAC Sanctions, and "
            "SAR Filing Requirements. Use this to find specific regulatory "
            "context, statutes, penalties, and compliance obligations relevant "
            "to a flagged transaction. CRITICAL: Use this tool EXACTLY ONCE per "
            "transaction. Do not loop or search multiple times."
        )
        args_schema: Type[BaseModel] = RegulatorySearchInput

        def _run(self, query: str) -> str:
            """Execute the regulatory search."""

            _step(f"RegulatorySearchTool invoked with: '{query[:80]}…'")

            # Try Azure first, fall back to local
            if AZURE_SEARCH_ENDPOINT and AZURE_SEARCH_KEY:
                return self._search_azure(query)
            else:
                return self._search_local(query)

        def _search_azure(self, query: str) -> str:
            """Vector search against Azure AI Search."""

            try:
                from openai import AzureOpenAI
                from azure.core.credentials import AzureKeyCredential
                from azure.search.documents import SearchClient
                from azure.search.documents.models import VectorizedQuery

                AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT", "")
                AZURE_OPENAI_KEY = os.getenv("AZURE_OPENAI_KEY", "")
                AZURE_OPENAI_EMBEDDING_DEPLOYMENT = os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT", "text-embedding-3-small")

                # Generate query embedding
                client = AzureOpenAI(
                    api_key=AZURE_OPENAI_KEY,
                    api_version="2023-05-15",
                    azure_endpoint=AZURE_OPENAI_ENDPOINT
                )
                response = client.embeddings.create(input=query, model=AZURE_OPENAI_EMBEDDING_DEPLOYMENT)
                query_vector = response.data[0].embedding

                # Execute vector search
                search_client = SearchClient(
                    endpoint=AZURE_SEARCH_ENDPOINT,
                    index_name=INDEX_NAME,
                    credential=AzureKeyCredential(AZURE_SEARCH_KEY),
                )

                vector_query = VectorizedQuery(
                    vector=query_vector,
                    k_nearest_neighbors=2,
                    fields="contentVector",
                )

                results = search_client.search(
                    search_text=query,  # Hybrid search: text + vector
                    vector_queries=[vector_query],
                    select=["title", "content", "category", "jurisdiction"],
                    top=2,
                )

                return self._format_results(results)

            except Exception as e:
                _step(f"Azure search failed: {e}. Falling back to local.")
                return self._search_local(query)

        def _search_local(self, query: str) -> str:
            """Cosine similarity search against local embedded documents."""

            try:
                import numpy as np
                from openai import AzureOpenAI

                AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT", "")
                AZURE_OPENAI_KEY = os.getenv("AZURE_OPENAI_KEY", "")
                AZURE_OPENAI_EMBEDDING_DEPLOYMENT = os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT", "text-embedding-3-small")

                _step("Using local regulatory knowledge base (Azure not configured)")

                # Load embedded documents
                if not os.path.exists(LOCAL_EMBEDDINGS_PATH):
                    return (
                        "ERROR: Regulatory knowledge base not found. "
                        "Run `python -m agents.setup_azure_search` first to "
                        "generate embeddings."
                    )

                with open(LOCAL_EMBEDDINGS_PATH, "r") as f:
                    documents = json.load(f)

                # Generate query embedding
                client = AzureOpenAI(
                    api_key=AZURE_OPENAI_KEY,
                    api_version="2023-05-15",
                    azure_endpoint=AZURE_OPENAI_ENDPOINT
                )
                response = client.embeddings.create(input=query, model=AZURE_OPENAI_EMBEDDING_DEPLOYMENT)
                query_vector = np.array(response.data[0].embedding)

                # Compute cosine similarity
                scored_docs = []
                for doc in documents:
                    doc_vector = np.array(doc["contentVector"])
                    similarity = np.dot(query_vector, doc_vector) / (
                        np.linalg.norm(query_vector) * np.linalg.norm(doc_vector)
                    )
                    scored_docs.append((similarity, doc))

                # Sort by similarity (descending) and take top 2
                scored_docs.sort(key=lambda x: x[0], reverse=True)
                top_docs = scored_docs[:2]

                # Format results
                results_text = []
                for score, doc in top_docs:
                    results_text.append(
                        f"[{doc['title']}] (Relevance: {score:.3f})\n"
                        f"Category: {doc['category']} | "
                        f"Jurisdiction: {doc['jurisdiction']}\n"
                        f"{doc['content']}\n"
                    )

                if not results_text:
                    return "No relevant regulatory documents found."

                return "\n---\n".join(results_text)

            except Exception as e:
                return f"ERROR searching regulatory knowledge base: {str(e)}"

        def _format_results(self, results) -> str:
            """Format Azure Search results into a readable string."""

            results_text = []
            for result in results:
                results_text.append(
                    f"[{result['title']}]\n"
                    f"Category: {result['category']} | "
                    f"Jurisdiction: {result['jurisdiction']}\n"
                    f"{result['content']}\n"
                )

            if not results_text:
                return "No relevant regulatory documents found."

            return "\n---\n".join(results_text)

    return RegulatorySearchTool()


# ─── Agent Definitions ────────────────────────────────────────────────────────

def create_agents(llm, regulatory_tool):
    """
    Create the three-agent Virtual Risk Committee.
    Returns (compliance_auditor, quant_analyst, cro).
    """

    from crewai import Agent

    _header("Creating Agents")

    # ── Agent 1: Compliance Auditor ────────────────────────────────────
    compliance_auditor = Agent(
        role="Senior Compliance Auditor",
        goal=(
            "Identify which specific financial regulations, statutes, or "
            "sanctions the flagged transaction may violate. Cite the exact "
            "rule, statute number, and potential penalties."
        ),
        backstory=(
            "You are a meticulous Senior Compliance Auditor with 15 years of "
            "experience at a Tier-1 investment bank. You specialize in BSA/AML, "
            "OFAC sanctions screening, and SEC enforcement actions. You have "
            "access to a curated regulatory knowledge base and always cite "
            "specific statutes and regulations in your findings. You never "
            "speculate — every claim must be backed by regulatory text. "
            "CRITICAL: You must execute the Regulatory Search Tool EXACTLY ONCE. "
            "Do not invoke the tool multiple times. Read the returned text "
            "and immediately formulate your answer."
        ),
        tools=[regulatory_tool],
        llm=llm,
        max_iter=3,
        max_execution_time=60,
        verbose=True,
        allow_delegation=False,
    )
    _step("Agent 1 — Compliance Auditor (RAG-equipped) ✔")

    # ── Agent 2: Quantitative Analyst ──────────────────────────────────
    quant_analyst = Agent(
        role="Quantitative Risk Analyst",
        goal=(
            "Evaluate the financial impact and risk exposure of the flagged "
            "transaction on the institution's portfolio. Assess velocity, "
            "concentration risk, counterparty exposure, and potential "
            "loss scenarios."
        ),
        backstory=(
            "You are a Quantitative Risk Analyst (CFA, FRM certified) at a "
            "major financial institution. You specialize in transaction "
            "pattern analysis, VaR calculations, and stress testing. You "
            "evaluate flagged transactions by analyzing the transaction "
            "amount, frequency, counterparty risk, geographic exposure, "
            "and historical account behavior to determine the potential "
            "financial impact. You present your analysis with precise "
            "numbers and risk metrics."
        ),
        llm=llm,
        max_iter=3,
        max_execution_time=60,
        verbose=True,
        allow_delegation=False,
    )
    _step("Agent 2 — Quantitative Analyst ✔")

    # ── Agent 3: Chief Risk Officer ────────────────────────────────────
    cro = Agent(
        role="Chief Risk Officer (CRO)",
        goal=(
            "Synthesize the compliance audit and quantitative analysis into "
            "a concise executive brief with exactly 3 bullet points and a "
            "clear recommendation. Then draft a Suspicious Activity Report "
            "(SAR) narrative."
        ),
        backstory=(
            "You are the Chief Risk Officer of a major financial institution, "
            "reporting directly to the Board of Directors. You receive the "
            "compliance audit (regulatory violations found) and the "
            "quantitative risk assessment (financial impact analysis) from "
            "your team. Your role is to make the final decision and produce "
            "two deliverables: (1) A 3-bullet executive summary with a "
            "clear action recommendation (Freeze Account, Allow Trade, or "
            "Request KYC Update), and (2) A draft Suspicious Activity "
            "Report (SAR) narrative suitable for FinCEN filing."
        ),
        llm=llm,
        max_iter=3,
        max_execution_time=60,
        verbose=True,
        allow_delegation=False,
    )
    _step("Agent 3 — Chief Risk Officer (CRO) ✔")

    return compliance_auditor, quant_analyst, cro


# ─── Task Definitions ─────────────────────────────────────────────────────────

def create_tasks(agents: tuple, transaction_data: dict):
    """
    Create the three sequential tasks for the Virtual Risk Committee.
    Returns (task_compliance, task_quant, task_cro).
    """

    from crewai import Task

    compliance_auditor, quant_analyst, cro = agents
    txn_json = json.dumps(transaction_data, indent=2)

    _header("Creating Tasks")

    # ── Task 1: Regulatory Compliance Check ────────────────────────────
    task_compliance = Task(
        description=(
            f"A transaction has been flagged by our ML anomaly detection model. "
            f"Analyze the following flagged transaction and use the Regulatory "
            f"Search Tool to identify ALL applicable regulations that may be "
            f"violated.\n\n"
            f"FLAGGED TRANSACTION DATA:\n"
            f"```json\n{txn_json}\n```\n\n"
            f"Your analysis MUST include:\n"
            f"1. The specific regulation(s) potentially violated (cite statute numbers).\n"
            f"2. The exact penalties and enforcement actions specified by those regulations.\n"
            f"3. Whether this transaction pattern matches known typologies "
            f"(structuring, wash trading, sanctions evasion, etc.).\n"
            f"4. The reporting obligations triggered (SAR, CTR, OFAC block report)."
        ),
        expected_output=(
            "A detailed compliance analysis with cited regulatory references, "
            "specific statute numbers, applicable penalties, matched typologies, "
            "and triggered reporting obligations."
        ),
        agent=compliance_auditor,
    )
    _step("Task 1 — Regulatory Compliance Check ✔")

    # ── Task 2: Quantitative Risk Assessment ───────────────────────────
    task_quant = Task(
        description=(
            f"Based on the compliance audit findings from the previous task, "
            f"perform a quantitative risk assessment of the following flagged "
            f"transaction.\n\n"
            f"FLAGGED TRANSACTION DATA:\n"
            f"```json\n{txn_json}\n```\n\n"
            f"Your analysis MUST include:\n"
            f"1. VELOCITY ANALYSIS: The account made {len(transaction_data.get('related_transactions', []))} "
            f"related transactions in rapid succession. Calculate the total "
            f"exposure and velocity rate.\n"
            f"2. CONCENTRATION RISK: Assess if this pattern indicates "
            f"concentrated exposure to a single counterparty or region.\n"
            f"3. DEVIATION FROM BASELINE: The account's historical average "
            f"transaction is ${transaction_data.get('account_metadata', {}).get('historical_avg_transaction', 0):,.2f}. "
            f"Calculate the standard deviation and z-score of the flagged amount.\n"
            f"4. POTENTIAL LOSS SCENARIO: Estimate the worst-case financial "
            f"impact if this activity continues undetected for 30 days."
        ),
        expected_output=(
            "A quantitative risk report with precise calculations including: "
            "total exposure amount, velocity metrics, deviation from baseline "
            "(z-score), concentration risk assessment, and worst-case loss "
            "estimate over 30 days."
        ),
        agent=quant_analyst,
    )
    _step("Task 2 — Quantitative Risk Assessment ✔")

    # ── Task 3: Executive Brief + SAR Draft ────────────────────────────
    task_cro = Task(
        description=(
            "You have received the Compliance Audit and Quantitative Risk "
            "Assessment from your team. Now produce two deliverables:\n\n"
            "DELIVERABLE 1 — EXECUTIVE BRIEF:\n"
            "Write exactly 3 concise bullet points summarizing:\n"
            "  • Bullet 1: The regulatory violation identified (with statute).\n"
            "  • Bullet 2: The quantified financial risk/exposure.\n"
            "  • Bullet 3: Your recommended action.\n"
            "End with a clear RECOMMENDATION from this list:\n"
            "  [FREEZE ACCOUNT] / [ALLOW TRADE] / [REQUEST KYC UPDATE] / [ESCALATE TO LEGAL]\n\n"
            "DELIVERABLE 2 — SAR NARRATIVE DRAFT:\n"
            "Draft a Suspicious Activity Report narrative following FinCEN "
            "guidelines. Include:\n"
            "  • Subject information (account holder, account details).\n"
            "  • Suspicious activity description (dates, amounts, pattern).\n"
            "  • Why the activity is suspicious (regulatory context).\n"
            "  • Actions taken by the institution.\n\n"
            "Format the output clearly with headers for each deliverable."
        ),
        expected_output=(
            "Two clearly formatted sections:\n"
            "1. EXECUTIVE BRIEF: Exactly 3 bullet points plus a bracketed "
            "recommendation.\n"
            "2. SAR NARRATIVE DRAFT: A complete suspicious activity report "
            "narrative suitable for FinCEN filing."
        ),
        agent=cro,
    )
    _step("Task 3 — Executive Brief + SAR Draft ✔")

    return task_compliance, task_quant, task_cro


# ─── Crew Assembly & Execution ────────────────────────────────────────────────

def run_risk_committee(transaction_data: Optional[dict] = None) -> str:
    """
    Assemble and run the Virtual Risk Committee crew.
    Returns the final output string.
    """

    from crewai import Crew, Process

    if transaction_data is None:
        transaction_data = SAMPLE_FLAGGED_TRANSACTION

    print("\n╔══════════════════════════════════════════════════════════╗")
    print("║   Veritas — Virtual Risk Committee                      ║")
    print("╚══════════════════════════════════════════════════════════╝")

    # ── Validate credentials ───────────────────────────────────────────
    _header("Initialization")

    if not GROQ_API_KEY or GROQ_API_KEY == "your-groq-api-key-here":
        print("  ✘ GROQ_API_KEY not configured in .env")
        print("    Get a free API key at: https://console.groq.com")
        sys.exit(1)

    _step(f"Groq model: {GROQ_MODEL}")
    _step(f"Embedding deployment: {AZURE_OPENAI_EMBEDDING_DEPLOYMENT}")
    _step(f"Azure Search: {'configured' if AZURE_SEARCH_ENDPOINT else 'local fallback'}")

    # ── Initialize LLM ────────────────────────────────────────────────
    _step("Initializing Groq LLM…")
    from crewai import LLM

    env_model = os.getenv("GROQ_MODEL", "groq/llama3-70b-8192")
    # Ensure it maps properly to the CrewAI LLM notation for groq correctly
    # If the prefix isn't there, CrewAI might get confused depending on version, 
    # but the instructions specifically indicate using LLM or ChatGroq properly.
    if not env_model.startswith("groq/"):
        env_model = f"groq/{env_model}"
        
    llm = LLM(
        model=env_model,
        api_key=GROQ_API_KEY,
        temperature=0.1,        # Low temperature for factual, precise output
    )
    _step("Groq LLM initialized ✔")

    # ── Build RAG tool ─────────────────────────────────────────────────
    _step("Building RegulatorySearchTool…")
    regulatory_tool = _build_regulatory_search_tool()
    _step("RegulatorySearchTool ready ✔")

    # ── Create agents ──────────────────────────────────────────────────
    agents = create_agents(llm, regulatory_tool)

    # ── Create tasks ───────────────────────────────────────────────────
    tasks = create_tasks(agents, transaction_data)

    # ── Assemble and run the crew ──────────────────────────────────────
    _header("Running Virtual Risk Committee")

    _step(f"Processing transaction: {transaction_data['transaction_id']}")
    _step(f"Anomaly type: {transaction_data.get('anomaly_type', 'unknown')}")
    _step(f"Amount: ${transaction_data['amount']:,.2f} {transaction_data['currency']}")
    print()

    crew = Crew(
        agents=list(agents),
        tasks=list(tasks),
        process=Process.sequential,
        verbose=True,
    )

    # ── Execute CrewAI (Isolate from Flask Threading if needed) ──
    try:
        result = crew.kickoff()
    except Exception as e:
        _step(f"Crew execution failed: {e}")
        raise e

    # ── Output ─────────────────────────────────────────────────────────
    _header("Virtual Risk Committee — Final Report")
    print()
    print(str(result))
    print()

    return str(result)


# ─── CLI Entry Point ──────────────────────────────────────────────────────────

def main():
    """Parse CLI arguments and run the risk committee."""

    parser = argparse.ArgumentParser(
        description="Veritas Virtual Risk Committee — Multi-Agent Advisory System",
    )
    parser.add_argument(
        "--transaction",
        type=str,
        default=None,
        help="JSON string of a flagged transaction (defaults to sample data)",
    )
    args = parser.parse_args()

    transaction_data = None
    if args.transaction:
        try:
            transaction_data = json.loads(args.transaction)
        except json.JSONDecodeError as e:
            print(f"  ✘ Invalid JSON in --transaction: {e}")
            sys.exit(1)

    run_risk_committee(transaction_data)


if __name__ == "__main__":
    main()
