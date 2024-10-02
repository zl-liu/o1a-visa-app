from fastapi import FastAPI, UploadFile, File
import pdfplumber
import os
import aiohttp
import asyncio
import csv
from datetime import datetime
import logging
import pandas as pd  # Import pandas for reading CSV
from chainlit.utils import mount_chainlit

# Initialize FastAPI app
app = FastAPI()

# Configure logging
logging.basicConfig(level=logging.INFO)

# Fetch OpenAI API key from environment variables
openai_api_key = os.getenv("OPENAI_API_KEY")
if not openai_api_key:
    raise ValueError("OpenAI API key not found in environment variables.")

# List of O-1A criteria
criteria = [
    "Awards",
    "Membership",
    "Press",
    "Judging",
    "Original Contribution",
    "Scholarly Articles",
    "Critical Employment",
    "High Remuneration"
]

# Load reference CSV
reference_sheet = pd.read_csv("reference_sheet.csv")  # Adjust the file path if necessary

# Helper function to extract text from PDF
def extract_text_from_pdf(file: UploadFile):
    with pdfplumber.open(file.file) as pdf:
        text = ''
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:  # Ensure valid text is extracted
                text += page_text
    return text

# Async helper function for making requests to OpenAI API via aiohttp
async def fetch_openai_completion(session, url, headers, prompt, pass_num, criterion):
    payload = {
        "model": "gpt-4",
        "messages": [
            {"role": "system", "content": "You are an expert in evaluating CVs for O-1A visa qualification."},
            {"role": "user", "content": prompt}
        ]
    }

    async with session.post(url, json=payload, headers=headers) as response:
        result = await response.json()
        content = result["choices"][0]["message"]["content"].strip()
        logging.info(f"Pass {pass_num}, Criterion: {criterion} - Done")
        return content

# First pass: Extract information for all criteria in parallel
async def batch_extract_for_criteria(cv_text):
    url = "https://api.openai.com/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {openai_api_key}",
        "Content-Type": "application/json"
    }

    # Generate prompts with additional data from the reference CSV
    prompts = []
    for criterion in criteria:
        row = reference_sheet[reference_sheet["Category"] == criterion]
        description = row["Description"].values[0] if not row.empty else "No description available"
        criteria_text = row["Criteria"].values[0] if not row.empty else "No criteria available"
        prompt = f"Here is the CV: {cv_text}. Extract relevant information about the applicant's {criterion}. " \
                 f"Use this USCIS information as a guide: Description: {description}, Criteria: {criteria_text}. Focus only on this aspect of the applicant, don't worry about information in the CV irrelevant to these."
        prompts.append(prompt)

    async with aiohttp.ClientSession() as session:
        tasks = [fetch_openai_completion(session, url, headers, prompt, pass_num=1, criterion=criterion) for
                 prompt, criterion in zip(prompts, criteria)]
        extractions = await asyncio.gather(*tasks)

    return dict(zip(criteria, extractions))

# Second pass: Evaluate evidence strength for all extracted information in parallel
async def batch_evaluate_evidence_strength(extractions):
    url = "https://api.openai.com/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {openai_api_key}",
        "Content-Type": "application/json"
    }

    # Generate prompts for evaluation, including additional information from the reference CSV
    prompts = []
    for criterion, extraction in extractions.items():
        row = reference_sheet[reference_sheet["Category"] == criterion]
        description = row["Description"].values[0] if not row.empty else "No description available"
        criteria_text = row["Criteria"].values[0] if not row.empty else "No criteria available"
        prompt = f"Evaluate the strength of this evidence for {criterion}: {extraction}. Description: {description}, Criteria: {criteria_text}. " \
                 "Focus only on this aspect of the applicant, don't worry about information in the CV irrelevant to these. After your assessment, provide a rating of [no evidence], [weak evidence], [reasonable evidence], or [remarkable evidence] " \
                 "on a new line STRICTLY in the format of 'OUTCOME_LABEL: [___]', fill ___ with either of the 4 choices."
        prompts.append(prompt)

    async with aiohttp.ClientSession() as session:
        tasks = [fetch_openai_completion(session, url, headers, prompt, pass_num=2, criterion=criterion) for
                 prompt, criterion in zip(prompts, criteria)]
        evaluations = await asyncio.gather(*tasks)

    # Extract the evidence rating from the structured response
    parsed_evaluations = {}
    for criterion, evaluation in zip(criteria, evaluations):
        # Find the label based on the specific format 'OUTCOME_LABEL: [___]'
        try:
            label_start = evaluation.lower().find('outcome_label:')
            if label_start != -1:
                label = evaluation[label_start:].split(":")[1].strip().lower()
                if "no evidence" in label:
                    parsed_evaluations[criterion] = "no evidence"
                elif "weak evidence" in label:
                    parsed_evaluations[criterion] = "weak evidence"
                elif "reasonable evidence" in label:
                    parsed_evaluations[criterion] = "reasonable evidence"
                elif "remarkable evidence" in label:
                    parsed_evaluations[criterion] = "remarkable evidence"
                else:
                    parsed_evaluations[criterion] = "no evidence"  # Default if format fails
        except:
            parsed_evaluations[criterion] = "no evidence"  # Default to no evidence in case of parsing issues

    return parsed_evaluations

# Write the first pass results to CSV for transparency
def write_first_pass_to_csv(extractions):
    filename = f"first_pass_results.csv"
    with open(filename, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Criterion", "Extraction"])
        for criterion, extraction in extractions.items():
            writer.writerow([criterion, extraction])

# Write the second pass results to CSV for transparency
def write_second_pass_to_csv(evaluations):
    filename = f"second_pass_results.csv"
    with open(filename, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Criterion", "Evaluation"])
        for criterion, evaluation in evaluations.items():
            writer.writerow([criterion, evaluation])

# Helper function to aggregate results
def aggregate_results(evaluations):
    # Define the sorting order for evidence strength
    evidence_ranking = {"no evidence": 0, "weak evidence": 1, "reasonable evidence": 2, "remarkable evidence": 3}

    # Sort by strongest criteria
    sorted_evaluations = sorted(evaluations.items(), key=lambda x: evidence_ranking[x[1]], reverse=True)

    # Consider the top 3 strongest criteria
    top_3 = [evaluation for criterion, evaluation in sorted_evaluations[:3]]

    # Determine overall qualification rating
    if top_3.count("remarkable evidence") >= 2:
        return "high"
    elif "remarkable evidence" in top_3 and top_3.count("reasonable evidence") >= 2:
        return "high"
    elif top_3.count("reasonable evidence") == 3:
        return "high"
    elif top_3.count("reasonable evidence") == 2 and "weak evidence" in top_3:
        return "medium"
    else:
        return "low"

# Route for processing CV PDF
@app.post("/assess_o1a/")
async def assess_o1a_cv(file: UploadFile = File(...)):
    # Extract text from uploaded PDF
    cv_text = extract_text_from_pdf(file)

    # First pass: Extract relevant information for each criterion in parallel
    extractions = await batch_extract_for_criteria(cv_text)

    # Write first pass results to CSV for transparency
    write_first_pass_to_csv(extractions)

    # Second pass: Evaluate the strength of the evidence in parallel
    evaluations = await batch_evaluate_evidence_strength(extractions)

    # Write second pass results to CSV for transparency
    write_second_pass_to_csv(evaluations)

    # Aggregate the results to determine the final qualification rating
    final_rating = aggregate_results(evaluations)

    # Return the extractions, evaluations, and final rating as a JSON response
    return {
        "extractions": extractions,
        "evaluations": evaluations,
        "final_rating": final_rating
    }

# Mount Chainlit as a sub-application
mount_chainlit(app=app, target="my_cl_app.py", path="/chainlit")

# Running with uvicorn: `uvicorn main:app --reload`
