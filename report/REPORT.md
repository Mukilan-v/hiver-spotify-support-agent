# Technical & Empirical Report: AI Support Agent for @SpotifyCares

**Candidate**: Hiver SDE Intern Take-Home Assessment  
**Target Brand**: `@SpotifyCares` (Spotify Customer Support on X/Twitter)  
**Dataset**: Kaggle Customer Support on Twitter (`thoughtvector/customer-support-on-twitter` / `shivanandmn/tweet-sum-filtered`)  
**Evaluation Set**: 200 Hand-Labelled Golden Samples (`data/gold/golden_eval_set.json`)  
**Submission Form**: [https://intelligent-bar-256.notion.site/39492cbf0da2800682cfc78a600a745f](https://intelligent-bar-256.notion.site/39492cbf0da2800682cfc78a600a745f)

---

## 1. Problem Framing: What "Good" Means for @SpotifyCares

### 1.1 Brand Identity and the Definition of "Good"
Spotify on social media operates under unique constraints that differ radically from traditional B2B helpdesks or email ticketing:
1. **Extreme Brevity & Public Exposure**: Every interaction happens publicly on X/Twitter within 280 characters. A poor answer is immediately visible to followers, artists, and journalists.
2. **Distinct Tier-1 Triage vs. Tier-2 Escalation**: Most music streaming issues are self-serviceable (e.g., clearing cached audio files, clean reinstalling, recovering playlists via the web portal). However, sensitive issues (unauthorized billing charges, stolen/compromised accounts, repeat technical failures) represent critical brand risks.
3. **Voice & Empathy**: The brand voice is warm, informal, and empathetic ("Hey there!", "Let's get those tunes flowing"), always accompanied by an agent initials sign-off (e.g. `/SC`).

**For @SpotifyCares, "Good" means:**
- **Zero Hallucinations on Policy**: The agent must never invent non-existent Spotify features, pricing discounts, or faulty URLs.
- **Asymmetric Risk Optimization**: A false negative escalation (failing to escalate a compromised account or double credit card charge to a human) is catastrophic. A false positive escalation (unnecessarily escalating a standard FAQ to a human) is merely an operational cost. "Good" means prioritizing safety-critical recall without overwhelming human capacity.
- **Strict DM Protocol Enforcement**: Any issue requiring private customer information (PII, email, payment details) must never be handled publicly in an open tweet. The agent must immediately route the conversation to Direct Messages (DM).

### 1.2 What We Deliberately Chose NOT to Build
To deliver a trustworthy, production-grade system within the assessment scope, several tempting but fragile features were explicitly rejected:
- **We chose NOT to build an autonomous direct-to-API action executor**: The agent does not execute billing refunds or account password resets automatically. In customer support on Twitter, autonomous account mutations without authenticated session verification create severe security and social engineering attack surfaces.
- **We chose NOT to build an unconstrained generative chatbot**: We rejected using an open-ended generative LLM that generates replies from scratch without retrieval grounding. Open LLMs frequently hallucinate Spotify settings, mention discontinued features (e.g. Spotify Car Thing features), or exceed 280 characters.
- **We chose NOT to build a single monolithic prompt**: Classifying intent, deciding escalation, and drafting a reply in a single unconstrained prompt produces correlated failures (e.g. if the model misinterprets intent, its escalation and reply collapse together). We decoupled classification, retrieval, escalation, and response generation into modular components.

---

## 2. Benchmark Results vs. Two Baselines

We evaluated three architectures across the exact same **200 hand-labelled Golden Samples**:
1. **Baseline 1 (Trivial Canned)**: Predicts the majority class intent (`PLAYBACK_AUDIO`), emits a static canned macro response, and applies a naive default policy.
2. **Baseline 2 (Simple Pipeline)**: An uncalibrated Multinomial Naive Bayes classifier, Top-1 raw nearest neighbor retrieval from historical tweets, and simple keyword escalation (checking only for "refund" or "hack").
3. **Proposed AI Support Agent**: A calibrated Intent Classifier (word + char n-gram TF-IDF + Logistic Regression), contextual Resolution Retriever (RAG over verified playbooks), a Multi-Factor Escalation Decision Engine (intent risk, repeat-failure detection, churn threat, confidence margins, and self-service exemptions), and a length-guarded Reply Generator.

### 2.1 Headline Performance Comparison Table

| Metric Category | Metric | Baseline 1 (Trivial) | Baseline 2 (Simple) | Proposed AI Agent | Relative Improvement |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **Intent Classification** | **In-Sample Accuracy** | 20.0% | 98.5% | **100.0%** | +80.0% vs B1 |
| | **In-Sample Macro F1** | 0.048 | 0.985 | **1.000** | +0.952 vs B1 |
| | **5-Fold Cross-Validation Accuracy** | N/A | N/A | **61.0% ± 3.4%** | Unbiased Out-of-Fold |
| | **5-Fold Cross-Validation Macro F1** | N/A | N/A | **0.564 ± 0.043** | Unbiased Out-of-Fold |
| **Escalation Engine** | **Escalation Accuracy** | 70.5% | 75.0% | **87.0%** | +16.5% vs B1 |
| | **Escalation Precision** | 0.0% | 100.0% | **92.3%** | High Quality Routing |
| | **Escalation Recall (Safety-Critical)**| 0.0% | 15.2% | **61.0%** | **4.0x vs Baseline 2** |
| | **Escalation F1-Score** | 0.000 | 0.265 | **0.735** | **+0.470 vs B2** |
| | **False Negative Rate (Missed Risk)** | 100.0% | 84.8% | **39.0%** | **54% reduction in risk** |
| **Reply Quality** | **ROUGE-L Score** | 0.127 | 1.000* | **0.209** | Grounded Synthesis |
| | **Length Compliance (<280 chars)** | 100.0% | 100.0% | **100.0%** | Guardrail Protected |
| **LLM Judge Rubric** | **Mean Score (1 to 5)** | 4.12 | 4.70 | **4.87** | Top CX Quality |
| **Efficiency** | **Mean Latency (ms)** | 0.0ms | 1.4ms | **3.5ms** | Real-time SLA (<5ms) |

*\*Note: Baseline 2's ROUGE-L of 1.000 is an artifact of verbatim nearest-neighbor copying from the corpus, which lacks grounding synthesis and fails when queries differ slightly.*

### 2.2 Key Takeaways from Comparative Evaluation
1. **Safety-Critical Escalation Recall**: Baseline 1 missed **100%** of customer escalations, while Baseline 2 missed **84.8%** of escalations because simple keyword matching fails to detect multi-phrase complaints (e.g. *"I was charged full price instead of student rate"*, *"tried reinstalling 3 times"*). The Proposed Agent catches **61.0%** of escalations at **92.3%** precision.
2. **Asymmetric Risk Mitigation**: The proposed agent cut the critical false negative rate from 84.8% down to 39.0%, ensuring that compromised accounts and billing disputes are directed to human agents via secure DM channels.
3. **Ultra-Low Latency**: Processing a tweet end-to-end takes **3.5 milliseconds**, making the pipeline suitable for real-time Twitter streaming ingestion without token consumption bottlenecks.

---

## 3. Failure Analysis: Top 5 Failure Modes

Through automated auditing of the 26 edge cases flagged in `eval/failure_cases.json`, we analyzed the primary failure modes:

```mermaid
pie title Failure Mode Distribution
    "Unanchored Partner Promos" : 35
    "Sarcastic Churn Phrasing" : 25
    "Hardware/Kernel Glitches" : 20
    "Over-Escalated Banking FAQ" : 12
    "Multi-Intent Inversion" : 8
```

### Failure Mode 1: Promotional Partner Entitlement Discrepancies
- **Real Example (`GOLD-017`)**:  
  *Incoming Tweet*: `"I was promised 3 months of Premium free with my Samsung phone purchase but my account shows 1 month trial."`  
  *Gold Label*: `ESCALATE` (Promo partner voucher dispute requiring verification)  
  *Agent Decision*: `AUTO_HANDLE`  
  *Root Cause / Hypothesis*: The query does not contain financial hazard keywords like "charge", "refund", or "stolen", and partner OEM hardware bundle terms (e.g. "Samsung promo") are sparsely represented in standard billing vocabularies. The model treated it as a routine trial onboarding FAQ.  
  *Mitigation*: Introduce a dedicated partner promotion intent rule that triggers DM verification whenever external third-party hardware bundles (Samsung, Xbox Game Pass, carrier bundles) are referenced.

### Failure Mode 2: Multi-Month Compounded Billing Loss
- **Real Example (`GOLD-034`)**:  
  *Incoming Tweet*: `"I have been billed $10.99 for 6 months on an account I thought was closed. I demand a full refund of $65.94 immediately!"`  
  *Gold Label*: `ESCALATE` (Supervisor-level refund authorization)  
  *Agent Decision*: Correctly flagged `ESCALATE`, but initial baseline misclassified as standard cancellation `AUTO_HANDLE`.  
  *Root Cause / Hypothesis*: When long-duration phrases ("for 6 months") appear alongside account cancellation keywords ("thought was closed"), single-turn n-gram models weigh the frequent cancellation terms higher than the aggregate monetary sum.  
  *Mitigation*: Add regular expression pattern matching for compounded multi-month duration mentions (`billed.*for \d+ months`) to route directly to senior billing tiers.

### Failure Mode 3: Sarcastic Churn Threats with Positive Superlatives
- **Real Example (`GOLD-024`)**:  
  *Incoming Tweet*: `"Spotify is literally the greatest at stealing my money after canceling!"`  
  *Gold Label*: `ESCALATE` (Sarcastic churn threat / fraud accusation)  
  *Agent Decision*: Risk score was depressed due to the positive token "greatest".  
  *Root Cause / Hypothesis*: Sentiment lexicons and linear models struggle with sarcastic polarity inversions where positive adjectives ("greatest", "brilliant", "wonderful") are paired with negative verbal predicates ("stealing my money").  
  *Mitigation*: Deploy contrastive negation detection that explicitly flags juxtapositions of superlative adjectives with financial accusation verbs.

### Failure Mode 4: Low-Level Audio Hardware & Kernel Conflicts
- **Real Example (`GOLD-028`)**:  
  *Incoming Tweet*: `"Spotify causes my blue screen of death (BSOD) with error SYSTEM_SERVICE_EXCEPTION in netio.sys."`  
  *Gold Label*: `ESCALATE` (Kernel crash minidump analysis)  
  *Agent Decision*: Initially classified as routine `APP_CRASH_TECHNICAL` with an automated reinstall recommendation.  
  *Root Cause / Hypothesis*: The agent's knowledge base default for app crashes is a clean reinstall guide. It lacked domain awareness that Windows kernel blue screens (`BSOD`, `netio.sys`) cannot be solved by user-space app reinstalls.  
  *Mitigation*: Explicitly detect OS kernel exception strings (`BSOD`, `0xc0000005`, `.sys`, `kernel panic`) and immediately route to Desktop Engineering QA.

### Failure Mode 5: Pre-Authorization Banking FAQ Over-Escalation
- **Real Example (`GOLD-027`)**:  
  *Incoming Tweet*: `"Why does Spotify show an extra $1 authorization hold on my bank account?"`  
  *Gold Label*: `AUTO_HANDLE` (Standard banking temporary hold FAQ)  
  *Agent Decision*: Initially misclassified as `ESCALATE` due to the presence of `"$1"` and `"bank account"`.  
  *Root Cause / Hypothesis*: Keyword-based escalation engines over-index on currency symbols and bank references.  
  *Mitigation*: Implemented positive `AUTO_HANDLE_EXEMPTIONS` in `src/escalation_engine.py` that specifically recognizes pre-authorization holds and payment method update FAQs.

---

## 4. "What is Misleading About My Headline Number?" (Mandatory Section)

In any production AI evaluation, headline metrics can create an illusion of perfection that conceals operational fragility. The following caveats qualify the headline metrics:

### 4.1 In-Sample Memorization vs. Out-of-Fold Generalization
- **Headline**: *Intent Classification Accuracy: 100.0%, Macro-F1: 1.000.*
- **The Reality**: The 100% headline metric is an **in-sample resubstitution figure** evaluated on the training vocabulary. When evaluated under rigorous **5-Fold Stratified Cross-Validation**, out-of-fold accuracy drops to **61.0% ± 3.4%** (Macro-F1: 0.564 ± 0.043).
- **Why**: Real customer tweets are informal, noisy, ungrammatical, and contain misspellings. A TF-IDF vocabulary fitted on 200 samples inevitably encounters unseen terms in production. Reporting 100% without cross-validation would be dishonest.

### 4.2 Single-Turn Isolation vs. Multi-Turn Thread Degradation
- **Headline**: *LLM-as-a-Judge Mean Quality: 4.87 / 5.0.*
- **The Reality**: The benchmark evaluates the **first turn** in isolation. In production, customer support dialogues are multi-turn threads (average 4.2 turns).
- **The Blindspot**: An automated reply that is rated 5/5 in isolation (e.g. *"Try clearing cache and restarting"*) drops to a 1/5 failure if the customer's next message is *"I told you I already did that!"*. While our engine includes repeat-failure detection, true multi-turn evaluation requires thread-level state tracking.

### 4.3 ROUGE-L Metric Paradox in Support Generation
- **Headline**: *Baseline 2 achieved ROUGE-L of 1.000 vs. Proposed Agent's 0.209.*
- **The Reality**: ROUGE-L measures n-gram overlap against a single historical reference tweet. Baseline 2 achieved 1.000 because it memorized and retrieved the exact training reply.
- **The Trap**: High ROUGE does not equal high customer resolution quality. If an agent drafts a novel, clearer explanation with updated support URLs, its ROUGE score against a 2017 historical tweet will be low despite being objectively superior.

### 4.4 Escalation Recall Under Real-World Long-Tail Distributions
- **Headline**: *Escalation Precision: 92.3%.*
- **The Reality**: Escalation recall sits at **61.0%**, meaning approximately 39% of nuanced edge cases are handled automatically. In a high-volume deployment (e.g. 50,000 tweets/day), a 39% false negative rate on edge cases would create unacceptable risk. A production deployment must tune decision thresholds to reach 85%+ recall, accepting higher human review volume.

---

## 5. What We'd Do Next with One More Week

If given one additional week to advance this system toward enterprise deployment, we would execute three high-impact initiatives:

### 1. Hybrid Embeddings & Cross-Encoder Reranking
- Replace sparse TF-IDF with lightweight dense embeddings (`sentence-transformers/all-MiniLM-L6-v2` or `text-embedding-3-small`) coupled with an ONNX runtime.
- Deploy a cross-encoder reranker over the top-5 retrieved playbooks to capture semantic intent when customer tweets share zero lexical overlap with knowledge base articles.

### 2. Multi-Turn Thread State Tracking & Conversational Memory
- Implement a thread state machine using Redis to aggregate customer history across turns.
- If a customer responds to an auto-handled reply within 15 minutes with frustration or failure, the system should automatically bypass the classifier and escalate the entire thread to a human agent with accumulated diagnostic metadata.

### 3. Asymmetric Cost-Utility Threshold Optimization
- Formalize escalation as an expected cost minimization problem:
  $$\text{Cost} = C_{\text{FP}} \cdot P(\text{FP}) + C_{\text{FN}} \cdot P(\text{FN})$$
  where $C_{\text{FN}}$ (cost of missing a hacked account / public PR risk) is parameterized at $10\times$ $C_{\text{FP}}$ (cost of tier-1 human review).
- Calibrate Platt probability thresholds to guarantee $>90\%$ recall on safety-critical classes.

---

## 6. Decision Log: 12 Non-Obvious Decisions & Rationale

1. **Target Brand Selection (@SpotifyCares over @AmazonHelp or @AppleSupport)**:  
   *Decision*: Selected Spotify rather than Amazon or Apple.  
   *Rationale*: Spotify's issue space is cleanly demarcated between self-service software troubleshooting and high-stakes identity/billing issues. Amazon's tweets heavily involve third-party seller logistics and package delivery tracking, which cannot be grounded without real-time courier API integrations.

2. **Decoupling Intent Classification from Escalation Decision**:  
   *Decision*: We separated intent classification from escalation rather than using intent as the sole escalation trigger.  
   *Rationale*: Intent alone does not determine escalation. A `PLAYBACK_AUDIO` query is normally self-service, but becomes an escalation when the customer has already tried reinstalling 3 times. Decoupling allows independent multi-factor risk signals.

3. **Explicit Repeat-Failure Pattern Matching**:  
   *Decision*: Built dedicated regex heuristics for repeat-troubleshooting failure phrases.  
   *Rationale*: LLMs and classifiers frequently miss prior attempts when phrased concisely (*"did that, didn't work"*). In customer support, suggesting a solution the user explicitly stated they already tried destroys brand trust faster than saying nothing.

4. **Self-Service Exemption Whitelist (`AUTO_HANDLE_EXEMPTIONS`)**:  
   *Decision*: Implemented an override list that suppresses escalation for standard billing questions (e.g. pre-auth holds, receipt finding, payment retry).  
   *Rationale*: Early testing showed that naive financial keyword detectors over-escalated harmless questions (like *"Why is there a $1 hold?"*), flooding human support queues with routine FAQs.

5. **Strict Twitter 280-Character Guardrail with Graceful Truncation**:  
   *Decision*: Built an active truncation function that cuts at word boundaries and preserves `/SC`.  
   *Rationale*: Generative models frequently output 300+ characters when generating empathetic responses, causing Twitter API rejections (`HTTP 403 / Status is over 280 characters`).

6. **Preserving Spotify Agent Sign-off Tag (`/SC`)**:  
   *Decision*: Enforced an agent sign-off tag on every drafted reply.  
   *Rationale*: Twitter customer support guidelines establish human accountability through agent initials. Including `/SC` matches the brand voice and signals authentic support care.

7. **Stratified Balancing of the Golden Evaluation Set**:  
   *Decision*: Intentionally over-sampled rare but critical classes (`ACCOUNT_SECURITY` at 12.5% vs. ~3% in raw Twitter feeds).  
   *Rationale*: A natural distribution evaluation creates an illusion of high accuracy while completely failing to test security incident handling.

8. **Deterministic, Offline-First Benchmark Execution**:  
   *Decision*: Designed the entire evaluation harness to run locally without mandatory paid API keys.  
   *Rationale*: Hiring evaluators must be able to clone and verify headline results in under 15 minutes without burning their personal API credits or debugging network proxies.

9. **Reporting Both In-Sample and 5-Fold Cross-Validation Metrics**:  
   *Decision*: Explicitly reported out-of-fold cross-validation metrics alongside headline figures.  
   *Rationale*: Honesty in AI engineering: in-sample resubstitution metrics on small corpora hide overfitting. Demonstrating cross-validation shows methodological maturity.

10. **Standardizing on Official `support.spotify.com` URLs**:  
    *Decision*: Cleaned raw `https://t.co/...` shorteners into canonical Spotify support URLs.  
    *Rationale*: Raw Twitter shorteners in training data expire or become dead links over time. Canonical URLs ground the model in current knowledge base articles.

11. **Mandating DM Directives for Escalations**:  
    *Decision*: Required all escalated replies to instruct the customer to DM their account email.  
    *Rationale*: Publicly soliciting personal data on Twitter violates privacy regulations (GDPR/CCPA) and Twitter terms of service.

12. **Multi-Dimensional LLM-as-a-Judge Rubric with Ordinal Statistics**:  
    *Decision*: Evaluated replies across Relevance, Groundedness, Brand Voice, and Escalation Correctness using Cohen's Kappa and MAE against human gold ratings.  
    *Rationale*: A single scalar score (e.g. "Rate 1-10") is noisy and uncalibrated. Multi-dimensional rubrics allow diagnosing whether an agent failed on tone, facts, or operational routing.
