# Design Choices and Output Evaluation

## 1. Overview of Design Choices

This document outlines the reasoning behind the architectural and operational design of a system that assesses O-1A visa qualifications based on an applicant's CV. The system is split into two distinct phases to ensure the accuracy and separation of concerns, allowing for modular, independent processing and evaluation. The backend architecture is designed to handle parallel processing and provide user-friendly interactions through a web-based interface.

In short, 

- Phase 1: 8 extraction calls, 1 per category.

- Phase 2: 8 evaluation calls, 1 per category. Phase 1 and Phase 2 are also guided by USCIS guidelines, "Appendix: Satisfying the O-1A Evidentiary Requirements"

- Phase 3: Aggregation and final assessment: Low, Medium or High.

Philosophy: Divide & Conquer. I personally choose this strategy based on past experience with detectors, extractors and evaluators. 

-Front end: Chainlit -- allows intuitive upload and supplies a chat-based interface

-Back end: FastAPI, handles the front end and calls to LLMs. 

-Extra: RAG. It helps with grounding the LLMs and helping users understand evidence behind the LLMs' judgement. 

Further improvements if I can work on this full-time:
- Fine-tuned and RLHF-ed dedicated models

- Tailored prompts

- Better RAG (many things can be experimented). For example, hierarchical semantic chunking.

- Better PDF parsing. There are many products and ideas in this domain.

## 2. Two-Phase System: Extractor and Evaluator

The system is built in two key phases:

### Phase 1: Extractor Phase

**Purpose**: The first pass focuses on extracting relevant information for each of the eight O-1A criteria:
1. Awards
2. Membership
3. Press
4. Judging
5. Original Contribution
6. Scholarly Articles
7. Critical Employment
8. High Remuneration

**Design**:
- In this phase, eight parallel API calls are made to the LLM (GPT-4), where each call is focused on extracting specific details about one criterion at a time.
- The LLM is guided to only pull out relevant details related to each specific criterion from the CV.

**Reason for Separation**:
- By splitting the process into individual criterion-based tasks, the LLM can focus on extracting only the data pertinent to one criterion at a time.
- This modularity reduces the cognitive load on the LLM, thus improving accuracy for each extraction.

**Example**: For the "Awards" category, the system sends only the CV and asks the LLM to extract information related to "Awards". This minimizes distractions and unrelated data processing.

### Phase 2: Evaluator Phase

**Purpose**: The second pass evaluates the strength of the evidence for each of the eight criteria based on the extractions from the first phase.

**Design**:
- Eight parallel API calls are again made, this time each call focusing on one of the extracted aspects from the first pass and evaluating how strong the evidence is.
- The LLM is tasked to judge the extracted information against a predefined scale: [no evidence], [weak evidence], [reasonable evidence], or [remarkable evidence].

**Separation of Evaluation**:
- The second phase ensures that extraction and evaluation are separated, contributing to better precision in both stages.
- The model can first focus on gathering all relevant details and then separately evaluate the strength of these details.

**Example**: If the system has extracted an award, the second call will evaluate how significant that award is based on specific criteria.

### Aggregation of Final Results

Once both phases are complete, the system aggregates the evaluation results into an overall assessment. This is done by looking at the top three criteria and their strength of evidence to determine the overall qualification rating.

**Aggregation Logic**:
1. Sort the eight categories based on their evaluation labels in the following order:
   - [no evidence]
   - [weak evidence]
   - [reasonable evidence]
   - [remarkable evidence]

2. The top three categories with the highest ranking labels are used to determine the overall assessment.

3. The system then uses the following rules for aggregation:

   **[High] Qualification**:
   - 3 [reasonable evidence] labels
   - 3 [remarkable evidence] labels
   - 2 [remarkable evidence] and 1 [reasonable evidence]
   - 2 [reasonable evidence] and 1 [remarkable evidence]

   **[Medium] Qualification**:
   - 2 [remarkable evidence] and 1 [minimal evidence]

   **[Low] Qualification**: All other combinations, including:
   - 1 [remarkable evidence] and 2 [minimal evidence]
   - 1 [reasonable evidence] and 2 [minimal evidence]
   - 2 [reasonable evidence] and 1 [minimal evidence]

This logic is inspired by USCIS's requirement that an O-1A visa applicant must show evidence in at least three distinct categories. By sorting and focusing on the top three categories, the system mimics the decision-making process often used by governmental agencies.

Additionally, government decision-making tends to favor categorical checks (i.e., confirming if three specific requirements are met) rather than applying a weighted sum. This informed our decision to prioritize categorical evidence over an arbitrary scoring system.

4. Through Chainlit, we also implement a RAG system that retrieves supporting text chunks from the CV that are relevant to each O1-A qualification category. This helps guide the LLMs, and help attorneys and applicants review the automatic process.

## 3. USCIS Guidelines Integration

To ensure that our system aligns with official USCIS requirements, we have incorporated the "Appendix: Satisfying the O-1A Evidentiary Requirements" from the USCIS Policy Manual (https://www.uscis.gov/policy-manual/volume-2-part-m#) into our `reference_sheet.csv` file.

### Purpose of the Reference Sheet
The `reference_sheet.csv` file serves as a crucial resource for both the Extractor and Evaluator phases. It contains detailed information about each of the eight O-1A criteria, including:

1. Category name
2. Official USCIS description
3. Specific criteria and examples provided by USCIS

### Integration in the System
- **Extractor Phase**: The LLM uses the descriptions and criteria from the reference sheet to accurately identify and extract relevant information from the CV for each category.
- **Evaluator Phase**: The LLM references the USCIS criteria to assess the strength of the extracted evidence, ensuring that the evaluation aligns with official USCIS standards.
- 
## 4. Backend Design: FastAPI and Parallel Processing

### Backend Architecture (FastAPI):
- The backend is built using FastAPI to handle incoming requests. FastAPI is highly scalable, supports asynchronous requests, and is well-suited for handling parallel LLM calls.
- Both the Extractor Phase and Evaluator Phase run in parallel. For each CV processed, eight LLM calls are made for the first phase and eight for the second phase. This helps ensure that all tasks are completed in a time-efficient manner without blocking the system.

### Front-End Design: Chainlit:
- Chainlit is used as the front-end interface to allow for a user-friendly experience where the user can upload a CV PDF.
- The web-based interface gives real-time feedback to the user, showing when their CV has been processed and displaying the overall qualification result.
- The Chainlit front end communicates with the FastAPI backend, sending the CV file, and receiving processed results and evaluation feedback.

## 5. How to Evaluate the System's Output

### Accuracy and Evaluation:
The system's output can be manually evaluated based on the accuracy of both the extractor phase and the evaluator phase. Key points of evaluation include:
- Correctness of extraction: Does the system correctly identify and extract all relevant details from the CV for each of the eight criteria?
- Precision of evaluation: Does the system correctly assign the strength of evidence for each of the extracted pieces of information?
- Does the final assessment, "Low", "Medium", and "High" makes sense to human experts?

### Improving Accuracy:
- Accuracy can be improved by refining the LLM prompts used in both the extraction and evaluation phases.
- Accuracy can be improved by fine-tuned or aligned (through RLHF) language models that are trained on relevant data where CVs are manually classified.

### Aggregation Evaluation:
- The system's final output is a clear overall rating based on the USCIS requirement of three distinct categories.
- It reflects how well the applicant qualifies for the O-1A visa based on their CV, and how convincing the extracted evidence is in each of the categories.
- This approach ensures that the system provides structured, transparent decisions, and these can be audited by reviewing the evidence provided for each of the eight categories.
