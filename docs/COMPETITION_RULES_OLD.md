# ALQAC 2026

> Obsolete reference snapshot. Use `docs/COMPETITION_RULES.md` for the latest BTC rules reflected in this repo.

**Automated Legal Question Answering Competition (ALQAC 2026)**  
Associated event of **KSE 2026**  
Held at **Kanazawa, Japan**  
**Date:** 11–14 November 2026

## Navigation

- Home
- Important Dates
- Leaderboard
- Program Committee
- Sponsorship
- Competition Registration
- Contact

## Overview

ALQAC 2026 introduces a single shared task on Vietnamese legal case understanding. Given a short case query derived from a Vietnamese court judgment, participating systems must predict whether the plaintiff or the defendant wins the case. In addition to the final prediction, systems are expected to retrieve supporting evidence from the case-content corpus and relevant legal provisions from the law corpus.

The task is designed to evaluate agentic legal AI systems that can combine case-level factual understanding, legal provision retrieval, evidence grounding, and outcome prediction. Instead of providing the full judgment directly to participants, the organizers expose segmented case content through official APIs. This setting encourages systems to actively search for relevant information, reason over retrieved evidence, and produce a verifiable prediction.

## Task Overview

For each test instance, participants are given a short natural-language query describing the dispute. The query includes the main parties, the disputed legal relationship or asset, a brief summary of the plaintiff's claim and the defendant's position, and a question asking whether the plaintiff or the defendant is more likely to win.

Participants must build a system that:

1. Reads the provided case query.
2. Calls the official Case Content API to retrieve relevant case segments.
3. Retrieves relevant legal provisions from the provided law corpus.
4. Predicts the final outcome of the case.
5. Submits the predicted outcome together with supporting case evidence and legal provisions.

The competition contains only one task: **Legal Case Outcome Prediction with Evidence Retrieval**.

The expected prediction is one of four labels:

| Label | Description |
|---|---|
| `A_WIN` | The court fully accepts all of the plaintiff's claims. |
| `PARTIAL_A_WIN` | The court partially accepts the plaintiff's claims, and the accepted portion is greater than 50%. |
| `PARTIAL_B_WIN` | The court partially accepts the plaintiff's claims, but the accepted portion is 50% or less. |
| `B_WIN` | The court fully rejects all of the plaintiff's claims. |

## Motivation

Legal case outcome prediction requires more than simple text classification. A strong system must understand the legal dispute, identify the claims of the parties, retrieve relevant factual evidence from the case record, retrieve applicable legal provisions, and reason about how the court is likely to resolve the dispute.

This task aims to encourage research on:

- Vietnamese legal judgment understanding.
- Retrieval-augmented legal reasoning.
- Agentic interaction with legal APIs.
- Evidence-grounded legal prediction.
- Vietnamese legal corpus retrieval.
- Transparent and verifiable legal AI systems.

The task setting reflects a realistic legal AI scenario: a system receives an initial case description, then must actively retrieve additional information before making a prediction.

## Data Resources

Participants will work with two main resources.

### Case Query Input

Each test case includes a short query generated from a Vietnamese court judgment. The query is intended to simulate the initial information given to a legal AI agent.

Example:

```json
{
  "case_id": "0001",
  "case_query": "Ông Nguyễn Khắc Vũ H1 (nguyên đơn) và Chu Quang Nguyễn H2 (bị đơn) tranh chấp hợp đồng chuyển nhượng quyền sử dụng đất đối với một phần thửa 366. Nguyên đơn yêu cầu được công nhận hợp đồng chuyển nhượng cho diện tích nêu trên. Agent cần dự đoán nguyên đơn thắng kiện hay bị đơn thắng kiện?"
}
```

The query does not reveal the court's reasoning, the final decision, or the winner of the case.

### 1. Case Content Corpus

The case content is segmented into smaller chunks and hosted by the organizers. Participants do not receive the full raw judgments directly for the test set. Instead, they must retrieve case segments through the official Case Content API.

Case segments may contain information such as:

- Plaintiff's claims.
- Defendant's arguments.
- Statements from related parties.
- Case facts.
- Procedural information.
- Court reasoning.
- Final verdict.

Participants are expected to call the API to identify the most relevant case segments for each query.

### 2. Law Corpus

The law corpus is provided to all participating teams. Teams may build their own retrieval system over this corpus to identify relevant legal provisions.

Each legal provision may include fields such as:

```json
{
  "law_id": "001abc",
  "content": [
    {
      "aid": "aaa",
      "content_Article": "Luật này quy định về việc thành lập, tổ chức, hoạt động, kiểm soát đặc biệt, tổ chức lại, giải thể tổ chức tín dụng;..."
    }
  ]
}
```

## Case Content API

The organizers will provide API access to the segmented case-content corpus. Participants should use this API to retrieve supporting evidence for each prediction.

### 1. Search Case Segments

**Endpoint:**

```http
POST /v1/case_segments/search
```

**Authentication:** Bearer token.

Request example:

```json
{
  "case_id": "001",
  "query": "hợp đồng chuyển nhượng quyền sử dụng đất thửa 396 nguyên đơn yêu cầu công nhận hợp đồng"
}
```

Response example:

```json
{
  "case_id": "0001",
  "result": {
    "hash_id": "hashsdjfvhlisduhfliudh",
    "text": "Ngày 04/5/2018 nguyên đơn có nhận chuyển nhượng của ông H2 diện tích 135m2..."
  }
}
```

### 2. API Usage

Participants may call the Case Content API multiple times for each test case. The number of API calls may be used for analysis or as a tie-breaker. The organizers may also define a maximum number of API calls per case or per submission.

## Input Format

The public test input will be released as a JSON file.

Example:

```json
[
  {
    "case_id": "0001",
    "case_query": "Ông Nguyễn Khắc Vũ H1 (nguyên đơn) và Chu Quang Nguyễn H2 (bị đơn) tranh chấp hợp đồng chuyển nhượng quyền sử dụng đất đối với một phần thửa 366. Nguyên đơn yêu cầu được công nhận hợp đồng chuyển nhượng cho diện tích nêu trên. Agent cần dự đoán nguyên đơn thắng kiện hay bị đơn thắng kiện"
  },
  {
    "case_id": "0002",
    "case_query": "..."
  }
]
```

Field descriptions:

| Field | Type | Description |
|---|---|---|
| `case_id` | string | Public identifier of the test case. |
| `case_query` | string | Short natural-language description of the dispute and the prediction question. |

The input file will not include the gold verdict, court reasoning, court decision, or gold evidence.

## Submission Details

### 1. Submission Format

Each team must submit a single JSON file named `submission.json`.

The submission must contain a list of predictions, one object per test case.

Example:

```json
[
  {
    "case_id": "0001",
    "prediction": "A_WIN",
    "law_evidence": ["270", "271", "357"],
    "case_evidence": ["hash1"],
    "api_calls": 30
  }
]
```

#### a. Required Fields

Each item in `law_evidence` must be a string legal-provision id from the law corpus. In this repository, submission export uses the article `aid` as a string, for example `"270"`.

#### b. Prediction Labels

The `prediction` field must be one of the following values:

- `A_WIN`
- `PARTIAL_A_WIN`
- `PARTIAL_B_WIN`
- `B_WIN`

If a case contains multiple claims, teams should focus on the main claim described in the `case_query`.

### 2. Documentation: Task & Scoring Rules

**Legal Case Outcome Prediction with Evidence Retrieval — ALQAC 2026**

#### 2.1 The Task

Given a short Vietnamese case query, participants must predict the case outcome and retrieve supporting case-content segments and relevant legal provisions to ground the prediction.

The outcome is one of four labels:

| Label | Meaning |
|---|---|
| `A_WIN` | Plaintiff wins. |
| `PARTIAL_A_WIN` | Plaintiff partially wins. |
| `B_WIN` | Defendant wins. |
| `PARTIAL_B_WIN` | Defendant partially wins. |

Outcome accuracy is exact-match across these four labels.

#### 2.2 How the Score Is Computed

| Weight | Component | Description |
|---:|---|---|
| 70% | **Outcome Accuracy** | Correct winner prediction. |
| 20% | **Penalized Case Recall** | Case-evidence recall multiplied by the API-efficiency factor. |
| 10% | **Micro Law F1** | Law-provision retrieval score over the full test set. |

The final score is computed as:

```text
FinalScore = 0.70 · OutcomeAccuracy + 0.20 · PenalizedCaseRecall + 0.10 · LawF1micro
```

#### 2.3 Outcome Accuracy

This component rewards systems that correctly predict whether the plaintiff or the defendant wins the case. The prediction must exactly match one of the four official labels.

#### 2.4 Case Evidence Recall

For each case, the system submits a set of case-content evidence segments. This component measures how many gold case evidence segments are successfully retrieved by the system.

#### 2.5 API Efficiency Penalty

The API call budget is case-dependent. Larger cases have more segments, so they are allowed more API calls.

The API-efficiency factor is:

```text
E_i = max(0, 1 − max(0, c_i − 2 · n_i) / (3 · n_i))
```

Where:

- `c_i` is the number of Case Content API calls used for case `i`.
- `n_i` is the number of segments in case `i`.

There is no penalty up to `2 · n_i` calls. The factor decays linearly to zero at `5 · n_i` calls.

#### 2.6 Penalized Case Recall

Penalized Case Recall combines case-evidence recall with the API-efficiency factor:

```text
PenalizedCaseRecall_i = CaseRecall_i · E_i
```

#### 2.7 Micro Law Evidence F1

Law evidence is evaluated using micro-averaged F1 over the full test set.

#### 2.8 Practical Interpretation

The metric is designed to reward systems that:

1. Predict the correct outcome.
2. Retrieve the correct case evidence.
3. Retrieve the relevant legal provisions.
4. Use the Case Content API efficiently.

### 3. Submission Validation

A submission may be rejected or partially ignored if it violates the required format.

The organizers may validate the following conditions:

- Every test case has exactly one submitted prediction.
- Every `case_id` exists in the official test set.
- There are no duplicate `case_id`s.
- The `prediction` value is one of the valid labels.
- `law_evidence` is a list of valid legal provision identifiers from the law corpus.
- The JSON file is valid and can be parsed automatically.

Duplicate evidence items may be automatically deduplicated before scoring.

### Example Submission

```json
[
  {
    "case_id": "0001",
    "prediction": "A_WIN",
    "law_evidence": [
      "001abc",
      "001aac"
    ]
  },
  {
    "case_id": "0002",
    "prediction": "B_WIN",
    "law_evidence": [
      "001abb",
      "001aba"
    ]
  }
]
```

### Extended Submission Format

A submission may also include optional case-evidence and API-call metadata:

```json
[
  {
    "case_id": "0001",
    "prediction": "A_WIN",
    "law_evidence": ["001abc"],
    "case_evidence": ["hash1"],
    "api_calls": 30
  }
]
```

Submission notes:

- Upload a single JSON array named `submission.json`, with one object per test case.
- Every test case needs exactly one prediction, with no duplicates.
- `case_id` must exist in the official test set.
- `prediction` must be one of `A_WIN`, `B_WIN`, `PARTIAL_A_WIN`, or `PARTIAL_B_WIN`.
- `law_evidence` contains law provision IDs and is required.
- `case_evidence` contains case segment IDs and is optional.
- `api_calls` records the number of Case Content API calls used and is optional.
- Omitting `case_evidence` and `api_calls` yields 0 on the 20% case-evidence component.
- Duplicate evidence IDs are de-duplicated before scoring.

## Restrictions

### 1. System Requirements and Reproducibility

Participating teams are encouraged to submit a short technical report describing their method, including:

- Retrieval strategy for the Case Content API.
- Retrieval strategy for the law corpus.
- Reasoning and prediction method.
- Models and tools used.
- Prompting or agent design, if applicable.
- Post-processing and validation steps.

The organizers may request source code, configuration files, or logs for verification and reproducibility.

### 2. Notes for Participants

- The case query is not sufficient to solve the task reliably. Systems should retrieve additional case segments through the official API.
- The law corpus is provided separately and should be used to retrieve relevant legal provisions.
- The `explanation` field is optional and is not the principal scoring component, but it may be used for qualitative analysis.
- The official ranking is based on the final score defined above.
- Participants should ensure that their submission file strictly follows the required JSON format.

### 3. Identity and Submission Limits

Teams are set up by the organizers, who issue each team a secret token. Keep it safe: it is shown only once and is required for every submission. The public leaderboard shows each team's best run.

Submissions are limited to **3 per team per day** on the official leaderboard.

### 4. Model Limitations

- **Proprietary Models Prohibited:** The use of closed or proprietary systems, including but not limited to ChatGPT, GPT-4, Claude, Gemini, or any other non-open API-based models, is strictly prohibited.
- **Model Size Limitation:** To ensure fairness and accessibility, and to encourage resource-conscious approaches, only open-weight models with fewer than 10 billion parameters are allowed. This levels the playing field for teams with limited computational resources.
- **External Dataset Limitations:** While querying online legal databases is permitted, the use of externally annotated datasets specifically created for legal question answering or legal entailment, such as pre-labeled QA pairs or entailment examples, is explicitly not allowed.

Any results or submissions obtained in violation of these rules will be disregarded in the final team ranking.

---

Photos from Unsplash  
Copyright © 2026 Nguyen Lab - JAIST.
