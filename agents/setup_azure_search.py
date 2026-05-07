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
        {
        "id": "reg-006",
        "title": "FATF 40 Recommendations — Anti-Money Laundering and Terrorism Financing",
        "category": "Global AML Standards",
        "jurisdiction": "International (FATF)",
        "content": (
            "The FATF Recommendations set out a comprehensive and consistent framework of measures which "
            "countries should implement in order to combat money laundering and terrorist financing. "
            "The 40 Recommendations guide companies to implement anti-fraud policies, build a proper AML program, "
            "practice accountability, and criminalize money laundering. Financial institutions "
            "must identify and assess the risks of potential breaches or evasion of targeted financial sanctions. "
            "Furthermore, if institutions suspect, or have reasonable grounds to suspect, that funds derive from criminal "
            "activity, they should report those suspicions promptly to the competent authorities."
        ),
    },
    {
        "id": "reg-007",
        "title": "FINRA Rules 3310 & 2090 — AML Compliance and KYC",
        "category": "AML / KYC",
        "jurisdiction": "United States (FINRA)",
        "content": (
            "FINRA Rule 3310 requires member organizations to develop and implement a written anti-money laundering "
            "program approved, in writing, by a member of senior management. Programs must establish policies "
            "to detect and report transactions required under the Bank Secrecy Act and provide for annual independent "
            "testing for compliance. FINRA Rule 2090 is a 'Know Your Customer' (KYC) requirement obligating "
            "broker-dealers to use reasonable diligence to learn essential facts about every customer. "
            "FINRA Rule 2090 non-compliance fines range from hundreds of thousands to millions of dollars based on severity."
        ),
    },
    {
        "id": "reg-008",
        "title": "Bank Secrecy Act (BSA) & Currency Transaction Reports",
        "category": "Reporting Obligations",
        "jurisdiction": "United States (FinCEN / Treasury)",
        "content": (
            "The Bank Secrecy Act (BSA) authorizes the Department of the Treasury to impose reporting requirements on "
            "financial institutions to help detect and prevent money laundering. Financial institutions must "
            "file reports of cash transactions exceeding $10,000 (daily aggregate amount) and report suspicious activity. "
            "Multiple transactions must be treated as a single transaction if they are by the same person and result in "
            "currency received or disbursed totaling more than $10,000 during any one business day. "
            "The Money Laundering Control Act of 1986 imposes criminal liability on a person or financial institution that "
            "knowingly assists in the laundering of money, or that structures transactions to avoid reporting them. "
            "A financial institution must file a Suspicious Activity Report on any transaction if it involves funds "
            "derived from illegal activity."
        ),
    },
        {
        "id": "reg-009",
        "title": "General Data Protection Regulation (GDPR) — EU Data Privacy",
        "category": "Data Privacy",
        "jurisdiction": "European Union (EU)",
        "content": (
            "The General Data Protection Regulation (GDPR) is a comprehensive data protection law "
            "enforceable across the European Union since May 25, 2018. KEY PRINCIPLES: (1) Lawfulness, "
            "fairness, and transparency — processing must be lawful, fair, and transparent to the data subject. "
            "(2) Purpose limitation — data collected for specified, explicit, and legitimate purposes. "
            "(3) Data minimization — only data necessary for the specific purpose may be collected. "
            "(4) Accuracy — personal data must be accurate and kept up to date. (5) Storage limitation — "
            "kept only as long as necessary. (6) Integrity and confidentiality — processed securely. "
            "(7) Accountability — controller must demonstrate compliance. DATA SUBJECT RIGHTS: Right to access, "
            "right to rectification, right to erasure (right to be forgotten), right to restrict processing, "
            "right to data portability, right to object, and rights related to automated decision-making. "
            "LAWFUL BASIS FOR PROCESSING: Consent, contract, legal obligation, vital interests, public task, "
            "or legitimate interests. BREACH NOTIFICATION: Controllers must notify supervisory authorities "
            "within 72 hours of becoming aware of a personal data breach. PENALTIES: Up to €20 million or "
            "4% of global annual turnover (whichever is higher) for infringements of core principles; "
            "up to €10 million or 2% for lesser violations. Cross-border data transfers require adequate "
            "safeguards such as Standard Contractual Clauses (SCCs) or Binding Corporate Rules (BCRs)."
        ),
    },
        {
        "id": "reg-010",
        "title": "Sarbanes-Oxley Act (SOX) — Corporate Financial Accountability",
        "category": "Financial Reporting",
        "jurisdiction": "United States (SEC)",
        "content": (
            "The Sarbanes-Oxley Act of 2002 (SOX) was enacted to protect investors by improving the accuracy "
            "and reliability of corporate disclosures. KEY PROVISIONS: Section 302 requires CEOs and CFOs to "
            "personally certify the accuracy of financial statements and adequacy of internal controls. "
            "Section 404 mandates management assessment of internal controls over financial reporting (ICFR) "
            "and requires independent auditor attestation. Section 409 requires real-time disclosure of material "
            "changes in financial condition. Section 802 imposes criminal penalties for destruction, alteration, "
            "or falsification of records in federal investigations. Section 806 provides whistleblower protections. "
            "Section 906 establishes criminal penalties for fraudulent financial reports: up to $5 million fine "
            "and 20 years imprisonment for willful violations. INTERNAL CONTROL REQUIREMENTS: Companies must "
            "maintain documented internal controls, perform regular testing, identify material weaknesses, "
            "and remediate deficiencies promptly. PCAOB Auditing Standard No. 2201 governs auditor evaluation "
            "of com"
            "pany internal controls. NON-COMPLIANCE CONSEQUENCES: Delisting from stock exchanges, "
            "inability to file required SEC reports, loss of investor confidence, and significant reputational damage."
        ),
    },
        {
        "id": "reg-011",
        "title": "Payment Card Industry Data Security Standard (PCI-DSS)",
        "category": "Cybersecurity",
        "jurisdiction": "International (PCI SSC)",
        "content": (
            "PCI-DSS is a set of security standards designed to ensure all companies that process, store, or "
            "transmit credit card information maintain a secure environment. VERSION 4.0 REQUIREMENTS: "
            "(1) Install and maintain network security controls (firewalls). (2) Apply secure configurations "
            "to all system components. (3) Protect stored account data — encryption, truncation, or tokenization. "
            "(4) Protect cardholder data with strong cryptography during transmission over open/public networks. "
            "(5) Protect all systems and networks from malicious software. (6) Develop and maintain secure systems "
            "and software. (7) Restrict access to system components by business need to know. (8) Identify and "
            "authenticate access to system components. (9) Restrict physical access to cardholder data. "
            "(10) Log and monitor all access to network resources and cardholder data. (11) Test security of "
            "systems and networks regularly. (12) Support information security with organizational policies "
            "and programs. COMPLIANCE LEVELS: Level 1 (6M+ transactions annually) requires annual QSA audit; "
            "Levels 2-4 may use Self-Assessment Questionnaire (SAQ). PENALTIES: Monthly fines from $5,000 to "
            "$100,000; potential termination of card processing privileges; liability for fraud losses. "
            "Data breach notification requirements vary by jurisdiction but typically require disclosure within "
            "72 hours to regulators and without undue delay to affected individuals."
        ),
    },
        {
        "id": "reg-012",
        "title": "Markets in Financial Instruments Directive II (MiFID II)",
        "category": "Securities Regulation",
        "jurisdiction": "European Union (ESMA)",
        "content": (
            "MiFID II is a legislative framework instituted by the European Union to regulate financial markets "
            "and improve investor protection, effective January 2018. SCOPE: Covers investment firms, market "
            "operators, data reporting service providers, and third-country firms providing services in the EU. "
            "KEY REQUIREMENTS: (1) Transaction reporting — firms must report complete and accurate details of "
            "transactions to competent authorities within T+1. (2) Best execution — firms must take all sufficient "
            "steps to obtain best possible result for clients considering price, costs, speed, likelihood of execution, "
            "and settlement. (3) Client categorization — clear distinction between retail, professional, and eligible "
            "counterparty clients with corresponding protection levels. (4) Product governance — manufacturers must "
            "identify target market and ensure products are distributed appropriately. (5) Inducements — strict rules "
            "on acceptance of fees, commissions, or non-monetary benefits. (6) Algorithmic trading — registration, "
            "testing, and kill-switch requirements. (7) Market abuse surveillance — enhanced surveillance for "
            "suspicious orders and transactions. (8) Telephone and electronic communications recording — record keeping "
            "of conversations and communications that result in transactions. PENALTIES: Administrative sanctions "
            "including public censure, fines up to €5 million or 10% of annual turnover for firms, disgorgement of "
            "profits, and withdrawal of authorization. Criminal penalties may apply for market abuse under MAR."
        ),
    },
        {
        "id": "reg-013",
        "title": "California Consumer Privacy Act (CCPA) / California Privacy Rights Act (CPRA)",
        "category": "Data Privacy",
        "jurisdiction": "United States (California)",
        "content": (
            "The CCPA (effective January 2020) and its amendment CPRA (effective January 2023) provide California "
            "consumers with comprehensive privacy rights. APPLICABILITY: Applies to for-profit businesses doing "
            "business in California that collect consumers personal information and meet thresholds: annual gross "
            "revenue over $25 million; buy/sell/share personal information of 100,000+ consumers/households; or "
            "derive 50%+ annual revenue from selling/sharing consumer personal information. CONSUMER RIGHTS: "
            "(1) Right to know — what personal information is collected, used, shared, or sold. (2) Right to delete "
            "personal information collected from the consumer (with exceptions). (3) Right to opt-out of sale or "
            "sharing of personal information. (4) Right to non-discrimination for exercising CCPA rights. "
            "(5) Right to correct inaccurate personal information (CPRA). (6) Right to limit use and disclosure of "
            "sensitive personal information (CPRA). (7) Right to data portability. SENSITIVE PERSONAL INFORMATION "
            "(CPRA): Social security numbers, precise geolocation, racial/ethnic origin, religious beliefs, "
            "genetic data, biometric information, health information, sex life/orientation, and contents of "
            "communications. REQUIRED DISCLOSURES: Privacy policy must describe categories of personal information "
            "collected, purposes, categories of third parties shared with, and consumers rights. PENALTIES: "
            "Civil penalties up to $2,500 per violation; $7,500 per intentional violation or violations involving "
            "minors. Private right of action for data breaches: statutory damages of $100-$750 per consumer per "
            "incident or actual damages, whichever is greater."
        ),
    },
        {
        "id": "reg-014",
        "title": "Dodd-Frank Wall Street Reform and Consumer Protection Act",
        "category": "Financial Regulation",
        "jurisdiction": "United States (CFTC/SEC)",
        "content": (
            "The Dodd-Frank Act of 2010 represents the most significant changes to financial regulation since "
            "the Great Depression. KEY PROVISIONS: (1) Volcker Rule — prohibits banks from proprietary trading "
            "and limits investments in hedge funds/private equity (Section 619). (2) Derivatives regulation — "
            "mandates central clearing of standardized swaps through regulated central counterparties (CCPs) and "
            "trading on regulated exchanges or swap execution facilities (SEFs). (3) Systemically Important "
            "Financial Institutions (SIFIs) — enhanced prudential standards for banks with $250B+ in assets or "
            "$10B+ in foreign exposure, including stress testing, living wills, and heightened capital requirements. "
            "(4) Consumer Financial Protection Bureau (CFPB) — created to regulate consumer financial products "
            "and services. (5) Whistleblower protections and rewards — SEC and CFTC whistleblower programs offering "
            "10-30% of monetary sanctions over $1 million. (6) Mortgage reform — ability-to-repay requirements, "
            "qualified mortgage standards, and restrictions on predatory lending. (7) Executive compensation — "
            "say-on-pay votes, clawback provisions for incentive-based compensation in case of accounting restatements. "
            "PENALTIES: Enhanced civil and criminal penalties for fraud, manipulation, and false reporting. "
            "CFTC civil monetary penalties up to $1 million per violation for individuals, $10 million for entities. "
            "SEC penalties include disgorgement, prejudgment interest, and civil penalties scaled to violation severity."
        ),
    },
        {
        "id": "reg-015",
        "title": "Health Insurance Portability and Accountability Act (HIPAA)",
        "category": "Healthcare Privacy",
        "jurisdiction": "United States (HHS/OCR)",
        "content": (
            "HIPAA establishes national standards to protect individuals medical records and other personal health "
            "information. COVERED ENTITIES: Health plans, healthcare clearinghouses, and healthcare providers who "
            "conduct certain electronic transactions. BUSINESS ASSOCIATES: Third parties performing functions on "
            "behalf of covered entities that involve protected health information (PHI). PRIVACY RULE: Standards "
            "for protecting PHI including limits on uses and disclosures without patient authorization, minimum "
            "necessary standard, and patient rights to access, amend, and receive accounting of disclosures. "
            "SECURITY RULE: Administrative safeguards (security management process, assigned security responsibilities, "
            "workforce training); physical safeguards (facility access controls, workstation security, device controls); "
            "technical safeguards (access controls, audit controls, integrity controls, transmission security). "
            "BREACH NOTIFICATION RULE: Covered entities must notify affected individuals within 60 days, Secretary "
            "of HHS (if 500+ individuals, immediately; if less, annually), and prominent media outlets (if 500+ "
            "individuals in a state/jurisdiction). ENFORCEMENT: Office for Civil Rights (OCR) investigates complaints "
            "and conducts compliance reviews. PENALTIES: Civil monetary penalties tiered by negligence level: "
            "Tier 1 (unknowing): $137-$68,928 per violation; Tier 2 (reasonable cause): $1,379-$68,928; Tier 3 "
            "(willful neglect, corrected): $13,785-$68,928; Tier 4 (willful neglect, not corrected): $68,928-$2,067,813 "
            "per violation. Annual maximums range from $25,000 to $2,067,813. Criminal penalties: up to $50,000 "
            "and 1 year imprisonment for wrongful disclosure; up to $100,000 and 5 years for false pretenses; "
            "up to $250,000 and 10 years for commercial gain or malicious harm."
        ),
    },
        {
        "id": "reg-016",
        "title": "Federal Information Security Management Act (FISMA) / NIST Cybersecurity Framework",
        "category": "Cybersecurity",
        "jurisdiction": "United States (Federal Government)",
        "content": (
            "FISMA requires federal agencies to develop, document, and implement information security programs. "
            "NIST SP 800-53 provides the security and privacy controls catalog. NIST CYBERSECURITY FRAMEWORK CORE "
            "FUNCTIONS: (1) Identify — develop organizational understanding to manage cybersecurity risk to systems, "
            "assets, data, and capabilities. (2) Protect — develop and implement appropriate safeguards to ensure "
            "delivery of critical services (access control, awareness training, data security, information protection "
            "processes, maintenance, protective technology). (3) Detect — develop and implement appropriate activities "
            "to identify occurrence of cybersecurity events (anomalies and events, continuous monitoring, detection "
            "processes). (4) Respond — develop and implement appropriate activities to take action regarding detected "
            "incidents (response planning, communications, analysis, mitigation, improvements). (5) Recover — develop "
            "and implement appropriate activities to maintain plans for resilience and restore capabilities impaired "
            "by incidents. RISK MANAGEMENT: Categorize information systems based on potential impact (low, moderate, "
            "high); select baseline security controls; implement controls; assess control effectiveness; authorize "
            "system operation based on acceptable risk; monitor security controls continuously. COMPLIANCE: Agencies "
            "must report security posture to OMB and DHS through CyberScope. Federal contractors handling controlled "
            "unclassified information (CUI) must comply with NIST SP 800-171. PENALTIES: Failure to comply can result "
            "in loss of federal funding, contract termination, False Claims Act liability, and debarment from future "
            "federal contracts."
        ),
    },
        {
        "id": "reg-017",
        "title": "Anti-Bribery and Corruption — UK Bribery Act 2010 & FCPA",
        "category": "Anti-Corruption",
        "jurisdiction": "United Kingdom / United States",
        "content": (
            "The UK Bribery Act 2010 and US Foreign Corrupt Practices Act (FCPA) represent the two most stringent "
            "anti-bribery regimes globally. UK BRIBERY ACT: Creates four offenses: (1) bribing another person "
            "(Section 1), (2) being bribed (Section 2), (3) bribery of foreign public officials (Section 6), and "
            "(4) failure of commercial organizations to prevent bribery by associated persons (Section 7). "
            "The Section 7 corporate offense has strict liability — no proof of knowledge or intent required. "
            "Adequate procedures defense available if organization can prove it had adequate procedures in place "
            "to prevent bribery. Penalties: unlimited fines for organizations; up to 10 years imprisonment for "
            "individuals. US FCPA: Prohibits bribery of foreign government officials to obtain or retain business. "
            "Anti-bribery provisions apply to issuers, domestic concerns, and certain foreign persons/entities. "
            "Accounting provisions require issuers to maintain accurate books/records and adequate internal controls. "
            "Penalties: criminal fines up to $2 million per violation for corporations; $100,000 for individuals; "
            "imprisonment up to 5 years. Under Alternative Fines Act, fines can reach twice the benefit sought. "
            "DOJ and SEC enforce FCPA through criminal prosecutions, civil injunctions, and administrative proceedings. "
            "COMPLIANCE PROGRAMS: Risk-based due diligence on third parties, gifts and hospitality policies, "
            "whistleblower mechanisms, regular training, and continuous monitoring are expected."
        ),
    },
        {
        "id": "reg-018",
        "title": "Modern Slavery Act 2015 — Supply Chain Transparency",
        "category": "Human Rights / Supply Chain",
        "jurisdiction": "United Kingdom",
        "content": (
            "The UK Modern Slavery Act 2015 requires organizations to be transparent about efforts to prevent "
            "modern slavery in their supply chains. SECTION 54 (TRANSPARENCY IN SUPPLY CHAINS): Commercial "
            "organizations carrying on business in the UK with annual turnover of £36 million or more must "
            "prepare a slavery and human trafficking statement each financial year. The statement must be approved "
            "by the board, signed by a director, and published on the organizations website with a prominent link. "
            "REQUIRED DISCLOSURES: (1) Organization structure, business, and supply chains. (2) Policies relating "
            "to slavery and human trafficking. (3) Due diligence processes in business and supply chains. "
            "(4) Risk assessment and management of slavery/human trafficking risks. (5) Key performance indicators "
            "to measure effectiveness of steps taken. (6) Training available to staff. MODERN SLAVERY OFFENSES: "
            "Slavery, servitude, forced/compulsory labor, and human trafficking carry maximum sentences of life "
            "imprisonment. ENFORCEMENT: Secretary of State may seek injunctions compelling compliance with Section 54. "
            "Non-compliance can result in unlimited fines for contempt of court. Reputational damage and exclusion "
            "from public procurement are significant business risks. GUIDANCE: Home Office guidance recommends "
            "six principles for effective supply chain transparency: embed responsible business practices, assess "
            "and manage risks, audit supply chain practices, train staff and suppliers, ensure access to remedy, "
            "and report annually on progress."
        ),
    },
        {
        "id": "reg-019",
        "title": "EMIR / Dodd-Frank — OTC Derivatives Clearing and Reporting",
        "category": "Derivatives Regulation",
        "jurisdiction": "European Union / United States",
        "content": (
            "The European Market Infrastructure Regulation (EMIR) and Dodd-Frank Title VII establish comprehensive "
            "regulation of over-the-counter (OTC) derivatives. CLEARING OBLIGATION: Standardized OTC derivatives "
            "must be cleared through Central Counterparties (CCPs). ESMA determines which classes are subject to "
            "mandatory clearing based on CCP authorization and risk management standards. MARGIN REQUIREMENTS: "
            "Uncleared derivatives require exchange of variation margin (VM) and initial margin (IM) to mitigate "
            "counterparty credit risk. Phase-in implementation based on average aggregate notional amount (AANA) "
            "thresholds. REPORTING OBLIGATION: All derivatives contracts must be reported to Trade Repositories (TRs) "
            "including counterparties, underlying instruments, notional amounts, maturities, and collateral. "
            "RISK MITIGATION: For non-cleared derivatives, firms must have risk management procedures including "
            "timely confirmation, portfolio reconciliation, portfolio compression, dispute resolution, and "
            "exchange of collateral. EXEMPTIONS: Intragroup transactions may be exempt subject to strict criteria. "
            "Pension fund exemption from clearing for certain interest rate and credit derivatives (time-limited). "
            "PENALTIES: Under EMIR, competent authorities can impose fines up to €5 million or twice the benefit "
            "gained/loss avoided. Under Dodd-Frank, CFTC civil penalties up to $1.606 million per violation for "
            "individuals ($16.065 million for entities), adjusted for inflation. Criminal penalties for willful "
            "violations include fines and imprisonment up to 10 years."
        ),
    },
        {
        "id": "reg-020",
        "title": "SFDR — Sustainable Finance Disclosure Regulation",
        "category": "ESG / Sustainable Finance",
        "jurisdiction": "European Union (EU)",
        "content": (
            "The Sustainable Finance Disclosure Regulation (SFDR) requires financial market participants to disclose "
            "how they integrate environmental, social, and governance (ESG) factors into their investment decisions. "
            "APPLICABILITY: Investment firms, insurance companies, pension funds, UCITS management companies, and "
            "AIFMs operating in the EU. PRODUCT CLASSIFICATION: Article 6 — products that integrate sustainability "
            "risks; Article 8 — products promoting environmental or social characteristics; Article 9 — products "
            "with sustainable investment as their objective. ENTITY-LEVEL DISCLOSURES: (1) Integration of sustainability "
            "risks in investment decision-making. (2) Consideration of principal adverse impacts (PAIs) of investment "
            "decisions on sustainability factors. (3) Remuneration policies consistent with ESG integration. "
            "PRODUCT-LEVEL DISCLOSURES: Pre-contractual disclosures, website disclosures, and periodic reporting "
            "on ESG characteristics/objectives, methodology, and data sources. PRINCIPAL ADVERSE IMPACTS (PAI): "
            "Mandatory reporting on 18 indicators including greenhouse gas emissions, biodiversity, water, waste, "
            "social and employee matters, respect for human rights, anti-corruption, and anti-bribery. TAXONOMY "
            "ALIGNMENT: For Article 8 and 9 products, disclosure of alignment with EU Taxonomy for environmentally "
            "sustainable activities including turnover, CapEx, and OpEx KPIs. PENALTIES: Member states designate "
            "competent authorities with powers to enforce including public statements, orders to cease conduct, "
            "administrative fines, and withdrawal of authorization. NCAs can impose fines up to €5 million or "
            "3% of annual turnover."
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
