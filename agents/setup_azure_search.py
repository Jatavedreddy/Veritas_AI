"""
Veritas — Azure AI Search Index Setup (RAG Knowledge Base)
═══════════════════════════════════════════════════════════
Creates and populates the `veritas-regulations` vector index
with mock regulatory documents for the Compliance Auditor Agent.

Documents include summaries of:
  1. Basel III Capital Requirements
  2. SEC Wash Trading Rules (Rule 10b-5)
  3. Bank Secrecy Act / AML Structuring Laws
  4. OFAC Sanctions Compliance Guidelines
  5. Suspicious Activity Report (SAR) Filing Requirements

Usage:
    python -m agents.setup_azure_search       # from project root
    python agents/setup_azure_search.py       # direct execution

Prerequisites:
    - Azure AI Search service provisioned
    - AZURE_SEARCH_ENDPOINT and AZURE_SEARCH_KEY set in .env
"""

import os
import sys
import time
from typing import Optional

from dotenv import load_dotenv

load_dotenv()

# ─── Configuration ─────────────────────────────────────────────────────────────

INDEX_NAME = "veritas-regulations"
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
AZURE_SEARCH_ENDPOINT = os.getenv("AZURE_SEARCH_ENDPOINT", "")
AZURE_SEARCH_KEY = os.getenv("AZURE_SEARCH_KEY", "")


# ─── Mock Regulatory Documents ────────────────────────────────────────────────

REGULATORY_DOCUMENTS = [
    {
        "id": "reg-001",
        "title": "Basel III — Minimum Capital Requirements",
        "category": "Capital Adequacy",
        "jurisdiction": "International (BIS)",
        "content": (
            "Basel III is a comprehensive set of reform measures developed by the "
            "Basel Committee on Banking Supervision (BCBS) to strengthen the regulation, "
            "supervision, and risk management of banks worldwide. Key provisions include: "
            "(1) Common Equity Tier 1 (CET1) capital ratio minimum of 4.5% of risk-weighted "
            "assets (RWA). (2) Total capital ratio minimum of 8% of RWA. (3) A mandatory "
            "capital conservation buffer of 2.5%, bringing the effective CET1 minimum to 7%. "
            "(4) A countercyclical buffer of 0-2.5% at discretion of national regulators. "
            "(5) Leverage ratio: minimum 3% Tier 1 leverage ratio. (6) Liquidity Coverage "
            "Ratio (LCR) requiring banks to hold sufficient high-quality liquid assets (HQLA) "
            "to cover net cash outflows over a 30-day stress period. (7) Net Stable Funding "
            "Ratio (NSFR) requiring available stable funding to exceed required stable funding "
            "over a one-year period. Banks found in violation of these capital adequacy "
            "requirements face supervisory actions including restrictions on dividend "
            "distributions and discretionary bonus payments."
        ),
    },
    {
        "id": "reg-002",
        "title": "SEC Rule 10b-5 — Prohibition of Wash Trading & Market Manipulation",
        "category": "Securities Fraud",
        "jurisdiction": "United States (SEC)",
        "content": (
            "Under Section 10(b) of the Securities Exchange Act of 1934 and SEC Rule 10b-5, "
            "it is unlawful for any person to employ any device, scheme, or artifice to defraud "
            "in connection with the purchase or sale of any security. WASH TRADING is specifically "
            "prohibited: this involves a trader simultaneously buying and selling the same "
            "financial instruments to create misleading, artificial trading activity. Key "
            "indicators of wash trading include: (1) Rapid round-trip transactions between "
            "related accounts with no net position change. (2) Trades executed at or near "
            "the same price within a short time window. (3) Pre-arranged trades between "
            "colluding parties. (4) Substantially identical buy and sell orders from the "
            "same beneficial owner. Penalties include: civil fines up to $5 million per "
            "violation for individuals, $25 million for entities, disgorgement of profits, "
            "injunctions, and potential criminal prosecution with imprisonment up to 20 years. "
            "Trade spoofing — placing orders with intent to cancel before execution — is a "
            "related violation under the Dodd-Frank Act Section 747."
        ),
    },
    {
        "id": "reg-003",
        "title": "Bank Secrecy Act (BSA) — Anti-Money Laundering & Structuring Laws",
        "category": "AML / KYC",
        "jurisdiction": "United States (FinCEN)",
        "content": (
            "The Bank Secrecy Act (BSA) of 1970, as amended by the USA PATRIOT Act, requires "
            "financial institutions to assist government agencies in detecting and preventing "
            "money laundering. STRUCTURING (also known as 'smurfing') is a federal crime under "
            "31 USC § 5324. It involves deliberately breaking up large cash transactions into "
            "multiple smaller transactions (typically just below the $10,000 Currency Transaction "
            "Report threshold) to evade BSA reporting requirements. Key provisions: (1) Financial "
            "institutions must file Currency Transaction Reports (CTRs) for cash transactions "
            "exceeding $10,000. (2) Structuring transactions to avoid CTR filing is a federal "
            "crime, even if the funds are legitimate. (3) Multiple transactions of $9,000-$9,999 "
            "from the same source within a short period are considered 'structuring indicators'. "
            "(4) Customer Due Diligence (CDD) rules require identifying beneficial owners of "
            "accounts. (5) Enhanced Due Diligence (EDD) is required for Politically Exposed "
            "Persons (PEPs) and high-risk jurisdictions. Penalties: criminal fines up to "
            "$500,000 and imprisonment up to 10 years per violation. Suspicious Activity "
            "Reports (SARs) must be filed for transactions involving $5,000 or more when "
            "the institution suspects money laundering, structuring, or other financial crimes."
        ),
    },
    {
        "id": "reg-004",
        "title": "OFAC Sanctions Compliance — SDN List & Geographic Restrictions",
        "category": "Sanctions",
        "jurisdiction": "United States (OFAC / Treasury)",
        "content": (
            "The Office of Foreign Assets Control (OFAC) administers and enforces economic "
            "and trade sanctions based on US foreign policy and national security goals. "
            "Financial institutions are STRICTLY PROHIBITED from processing transactions "
            "involving: (1) Specially Designated Nationals (SDNs) — individuals and entities "
            "on the OFAC SDN List. (2) Comprehensively sanctioned countries and regions: "
            "North Korea (DPRK), Iran, Syria, Cuba, and the Crimea region of Ukraine. "
            "(3) Sectoral sanctions programs targeting specific sectors in Russia, Venezuela, "
            "and other jurisdictions. (4) Any transaction designed to evade or circumvent "
            "sanctions. COMPLIANCE REQUIREMENTS: All financial institutions must implement "
            "a risk-based sanctions compliance program including: SDN list screening of all "
            "customers and counterparties, geographic risk assessment, real-time transaction "
            "monitoring, and escalation procedures for blocked or rejected transactions. "
            "Violations carry SEVERE PENALTIES: civil fines up to $356,579 per violation "
            "(adjusted for inflation) or twice the transaction value, and criminal penalties "
            "up to $20 million and 30 years imprisonment. SWIFT wire transfers to sanctioned "
            "regions must be automatically blocked and reported to OFAC within 10 business days."
        ),
    },
    {
        "id": "reg-005",
        "title": "Suspicious Activity Report (SAR) Filing Requirements — FinCEN Guidelines",
        "category": "Reporting Obligations",
        "jurisdiction": "United States (FinCEN)",
        "content": (
            "Under the Bank Secrecy Act and FinCEN regulations (31 CFR 1020.320), financial "
            "institutions are required to file Suspicious Activity Reports (SARs) when they "
            "detect known or suspected violations of law, or suspicious transactions relevant "
            "to money laundering, terrorist financing, or other financial crimes. FILING "
            "THRESHOLDS: (1) Banks must file a SAR for any transaction or pattern of "
            "transactions of $5,000 or more that the institution knows, suspects, or has "
            "reason to suspect involves funds from illegal activity or is designed to evade "
            "BSA reporting requirements. (2) For transactions involving insiders (directors, "
            "officers, employees), the threshold is $25,000. SAR CONTENT REQUIREMENTS: "
            "The report must include: (a) A clear narrative describing the suspicious "
            "activity. (b) Transaction details — dates, amounts, account numbers, and "
            "involved parties. (c) The reason(s) why the activity is suspicious. (d) Any "
            "actions taken by the institution. TIMING: SARs must be filed within 30 calendar "
            "days of initial detection of suspicious activity. If no suspect is identified, "
            "the deadline extends to 60 days. SARs are CONFIDENTIAL — institutions are "
            "prohibited from notifying the subject of the SAR (tipping off). Failure to "
            "file SARs can result in civil penalties of up to $1 million per violation."
        ),
    },
]


# ─── Utilities ─────────────────────────────────────────────────────────────────

def _header(title: str) -> None:
    width = 60
    print(f"\n{'─' * width}")
    print(f"  {title}")
    print(f"{'─' * width}")


def _step(msg: str) -> None:
    print(f"  ▸ {msg}")


# ─── Step 1: Generate Embeddings ──────────────────────────────────────────────

def generate_embeddings(documents: list[dict]) -> list[dict]:
    """Generate vector embeddings for each document using sentence-transformers."""

    _header("STEP 1 — Generating Embeddings")

    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        print("  ✘ sentence-transformers not installed.")
        print("    Run: pip install sentence-transformers")
        sys.exit(1)

    _step(f"Loading model: {EMBEDDING_MODEL}")
    model = SentenceTransformer(EMBEDDING_MODEL)
    embedding_dim = model.get_sentence_embedding_dimension()
    _step(f"Embedding dimensions: {embedding_dim}")

    for doc in documents:
        # Combine title + content for richer embeddings
        text = f"{doc['title']}. {doc['content']}"
        embedding = model.encode(text).tolist()
        doc["contentVector"] = embedding
        _step(f"Embedded: {doc['title'][:50]}… ({len(embedding)} dims)")

    return documents


# ─── Step 2: Create Azure Search Index ────────────────────────────────────────

def create_search_index(embedding_dim: int = 384) -> None:
    """Create the vector search index on Azure AI Search."""

    _header("STEP 2 — Creating Azure AI Search Index")

    if not AZURE_SEARCH_ENDPOINT or not AZURE_SEARCH_KEY:
        print("  ⚠ Azure credentials not configured.")
        print("    Set AZURE_SEARCH_ENDPOINT and AZURE_SEARCH_KEY in .env")
        print("    Skipping index creation (embeddings saved locally).")
        return False

    try:
        from azure.core.credentials import AzureKeyCredential
        from azure.search.documents.indexes import SearchIndexClient
        from azure.search.documents.indexes.models import (
            SearchIndex,
            SearchField,
            SearchFieldDataType,
            SimpleField,
            SearchableField,
            VectorSearch,
            HnswAlgorithmConfiguration,
            VectorSearchProfile,
        )
    except ImportError:
        print("  ✘ azure-search-documents not installed.")
        print("    Run: pip install azure-search-documents azure-core")
        sys.exit(1)

    _step(f"Connecting to: {AZURE_SEARCH_ENDPOINT}")

    index_client = SearchIndexClient(
        endpoint=AZURE_SEARCH_ENDPOINT,
        credential=AzureKeyCredential(AZURE_SEARCH_KEY),
    )

    # ── Vector search configuration ────────────────────────────────────
    vector_search = VectorSearch(
        algorithms=[
            HnswAlgorithmConfiguration(name="veritas-hnsw"),
        ],
        profiles=[
            VectorSearchProfile(
                name="veritas-vector-profile",
                algorithm_configuration_name="veritas-hnsw",
            ),
        ],
    )

    # ── Index schema ───────────────────────────────────────────────────
    fields = [
        SimpleField(
            name="id",
            type=SearchFieldDataType.String,
            key=True,
            filterable=True,
        ),
        SearchableField(
            name="title",
            type=SearchFieldDataType.String,
            analyzer_name="en.microsoft",
        ),
        SearchableField(
            name="content",
            type=SearchFieldDataType.String,
            analyzer_name="en.microsoft",
        ),
        SimpleField(
            name="category",
            type=SearchFieldDataType.String,
            filterable=True,
            facetable=True,
        ),
        SimpleField(
            name="jurisdiction",
            type=SearchFieldDataType.String,
            filterable=True,
            facetable=True,
        ),
        SearchField(
            name="contentVector",
            type=SearchFieldDataType.Collection(SearchFieldDataType.Single),
            searchable=True,
            vector_search_dimensions=embedding_dim,
            vector_search_profile_name="veritas-vector-profile",
        ),
    ]

    index = SearchIndex(
        name=INDEX_NAME,
        fields=fields,
        vector_search=vector_search,
    )

    _step(f"Creating/updating index: {INDEX_NAME}")
    result = index_client.create_or_update_index(index)
    _step(f"Index '{result.name}' ready ✔")

    return True


# ─── Step 3: Upload Documents ─────────────────────────────────────────────────

def upload_documents(documents: list[dict]) -> None:
    """Upload embedded documents to the Azure AI Search index."""

    _header("STEP 3 — Uploading Documents")

    if not AZURE_SEARCH_ENDPOINT or not AZURE_SEARCH_KEY:
        print("  ⚠ Azure credentials not configured — skipping upload.")
        _save_local_fallback(documents)
        return

    try:
        from azure.core.credentials import AzureKeyCredential
        from azure.search.documents import SearchClient
    except ImportError:
        print("  ✘ azure-search-documents not installed.")
        sys.exit(1)

    search_client = SearchClient(
        endpoint=AZURE_SEARCH_ENDPOINT,
        index_name=INDEX_NAME,
        credential=AzureKeyCredential(AZURE_SEARCH_KEY),
    )

    _step(f"Uploading {len(documents)} documents to index '{INDEX_NAME}'")

    result = search_client.upload_documents(documents=documents)

    succeeded = sum(1 for r in result if r.succeeded)
    failed = sum(1 for r in result if not r.succeeded)

    _step(f"Succeeded: {succeeded}  |  Failed: {failed}")

    if failed > 0:
        for r in result:
            if not r.succeeded:
                print(f"    ✘ {r.key}: {r.error_message}")


def _save_local_fallback(documents: list[dict]) -> None:
    """Save documents with embeddings locally as a fallback for dev/testing."""

    import json

    fallback_dir = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "..", "data", "processed"
    )
    os.makedirs(fallback_dir, exist_ok=True)

    fallback_path = os.path.join(fallback_dir, "regulatory_embeddings.json")

    # Convert numpy arrays to lists for JSON serialization
    docs_serializable = []
    for doc in documents:
        doc_copy = dict(doc)
        if "contentVector" in doc_copy:
            vec = doc_copy["contentVector"]
            doc_copy["contentVector"] = (
                vec if isinstance(vec, list) else vec.tolist()
            )
        docs_serializable.append(doc_copy)

    with open(fallback_path, "w") as f:
        json.dump(docs_serializable, f, indent=2)

    _step(f"Local fallback saved → {fallback_path}")
    _step("This file will be used by RegulatorySearchTool when Azure is unavailable.")


# ─── Main Pipeline ─────────────────────────────────────────────────────────────

def main() -> None:
    """Execute the full index setup pipeline."""

    print("\n╔══════════════════════════════════════════════════════════╗")
    print("║   Veritas — Azure AI Search Index Setup                 ║")
    print("╚══════════════════════════════════════════════════════════╝")

    # 1. Generate embeddings
    documents = generate_embeddings(REGULATORY_DOCUMENTS)

    # Determine embedding dimensions from the first document
    embedding_dim = len(documents[0]["contentVector"])

    # 2. Create/update the index
    index_created = create_search_index(embedding_dim)

    # 3. Upload documents
    upload_documents(documents)

    if index_created:
        print(f"\n  🚀 Index '{INDEX_NAME}' ready with {len(documents)} documents.")
    else:
        print(f"\n  📁 Embeddings generated and saved locally (Azure not configured).")

    print("  Ready for the Compliance Auditor Agent.\n")


# ─── Entry Point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    main()
