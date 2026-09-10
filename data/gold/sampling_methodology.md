# Golden Evaluation Dataset: Sampling & Labeling Methodology

## 1. Dataset Overview & Provenance

This golden evaluation dataset contains **200 hand-labelled examples** derived from the public Kaggle / Twitter Customer Support corpus (`thoughtvector/customer-support-on-twitter` and the curated `shivanandmn/tweet-sum-filtered` multi-turn dialogue subset).

The target brand for this evaluation is **@SpotifyCares**, Spotify's official customer care presence on X/Twitter.

### Provenance Pipeline:
1. **Extraction**: From 1,100 multi-turn customer care threads, all conversations involving `@SpotifyCares` (63 full dialogues, comprising 623 raw tweets and 267 customer-agent message pairs) were parsed and filtered.
2. **De-anonymization & Normalization**: Anonymized customer handle tokens (e.g. `@135060`, `@378966`) and raw URL shorteners (`https://t.co/...`) were cleaned into realistic incoming queries and official target Spotify support links (`https://support.spotify.com/...`).
3. **Stratified Balancing**: To test the agent across operational edge cases rather than only high-frequency easy queries, the dataset was stratified across 7 discrete customer intents, ensuring adequate representation of low-frequency, high-risk classes (such as `ACCOUNT_SECURITY`).

---

## 2. Intent Taxonomy & Stratification Distribution

The 200 examples are stratified into seven mutually exclusive, operational intents:

| Intent Class | Count | Pct (%) | Primary Operational Scope |
| :--- | :---: | :---: | :--- |
| **`PLAYBACK_AUDIO`** | 40 | 20.0% | Stuttering/skipping, song pausing, offline playback sync, greyed-out tracks, sound volume normalization. |
| **`BILLING_SUBSCRIPTION`** | 35 | 17.5% | Duplicate charges, student discount re-verification, family plan address validation, cancellation, pricing discrepancies. |
| **`APP_CRASH_TECHNICAL`** | 35 | 17.5% | Crash on launch, infinite loading spinner, Windows/Mac desktop freeze, black screen after updates. |
| **`ACCOUNT_SECURITY`** | 25 | 12.5% | Hacked accounts, unauthorized email changes, foreign logins, password reset email failures, phishing reports. |
| **`DEVICE_CONNECTIVITY`** | 25 | 12.5% | Bluetooth drops, CarPlay / Android Auto glitches, Alexa / Echo Dot integration, PS5 / console Connect issues. |
| **`LIBRARY_CONTENT`** | 25 | 12.5% | Accidentally deleted playlists, missing albums/tracks, local MP3 Wi-Fi sync, playlist folder management. |
| **`FEEDBACK_CHITCHAT`** | 15 | 7.5% | Feature requests (heart button, HiFi), UI design critiques, brand compliments, community engagement. |
| **Total** | **200** | **100.0%** | |

---

## 3. Escalation Labeling Guidelines

Every example is assigned an operational action label:
- **`AUTO_HANDLE`** (130 examples, 65.0%):
  The agent can independently resolve the ticket without human intervention because:
  1. The issue is covered by a verified self-service resolution flow (e.g. clearing cache, clean reinstall, web portal playlist recovery).
  2. The issue represents standard platform policy or copyright reality (e.g. licensing expirations causing greyed-out songs, family plan shared address rule).
  3. No private customer records, credit card numbers, or backend administrative overrides are required.
- **`ESCALATE`** (70 examples, 35.0%):
  The agent MUST route the ticket to a human support specialist because:
  1. **Financial / Payment Gateways**: Duplicate credit card charges, refund requests, unverified bank holds.
  2. **Security & Identity Takeover**: Customer account compromised, email altered without permission, session hijacking.
  3. **Exhausted Self-Service (Repeat Failure)**: The customer explicitly stated that they already performed standard steps (e.g., *"I already clean reinstalled and restarted twice and it still crashes"*). Repeating automated advice in this scenario guarantees customer rage.
  4. **High Churn / Legal Threat**: Extreme negative sentiment with threats to cancel or switch to competitors (e.g. Apple Music).
  5. **Hardware / Operating System Bug**: Rare memory crashes, kernel panics, or unhandled device conflicts requiring QA bug logging.

---

## 4. Edge Cases and Adversarial Samples

To prevent benchmark inflation, 25% of the evaluation set consists of deliberate edge cases:
1. **`repeat_failure` (8 examples)**: Customer explicitly states prior troubleshooting steps failed. A naive classifier or keyword bot suggests reinstalling again; our golden escalation requires escalating to Human QA.
2. **`pii_financial` (12 examples)**: Involves credit cards, invoices, or hacked credentials where the agent must migrate to DM immediately to protect user privacy.
3. **`sarcasm_churn` (10 examples)**: Highly sarcastic or frustrated messages threatening churn (*"Spotify is literally the greatest at stealing my money after canceling"*). Naive sentiment models misclassify "greatest" as positive; golden escalation flags churn risk.
4. **`multi_intent` (10 examples)**: Customer asks about billing and a playback issue in the same tweet. Labeled with the dominant operational risk intent (`BILLING_SUBSCRIPTION`).

---

## 5. Human Judge Ground Truth Rubric (1-5 Scale)

Each example was independently annotated across four dimensions:
1. **Relevance (1-5)**: Does the draft reply directly answer the customer's specific question?
2. **Groundedness (1-5)**: Is the suggested resolution factually supported by historical brand playbooks and Spotify knowledge base articles?
3. **Brand Voice (1-5)**: Is the reply warm, concise, within Twitter's 280-character limit, empathetic, and signed off with Spotify's signature agent sign-off (`/SC` or agent initials)?
4. **Escalation Correctness (1-5)**: Was the decision to auto-handle vs. escalate operationally correct, and was the stated reason valid?

This human ground truth rubric serves as the calibration baseline for evaluating the LLM-as-a-judge system in `eval/human_agreement.py`.
