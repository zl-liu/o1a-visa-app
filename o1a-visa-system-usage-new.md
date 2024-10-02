## Usage

There are two ways to use the O-1A Visa Qualification System:

### 1. Using the Chainlit Web Interface

1. Open your browser and go to the Chainlit app at:

   ```
   http://0.0.0.0:8500
   ```

2. Upload a resume/CV PDF file when prompted.

3. Wait for the system to process the file.

4. After processing, the system will output the likelihood of success for the O-1A visa application as either [Low], [Medium], or [High].

### 2. Directly Hitting the API Endpoint

Alternatively, you can interact with the system directly through its API endpoint. Here's an example using curl:

```bash
curl -X 'POST' \
  'http://127.0.0.1:8000/assess_o1a/' \
  -H 'accept: application/json' \
  -H 'Content-Type: multipart/form-data' \
  -F 'file=@Zhengliang_Liu_Resume.pdf'
```

Replace `Zhengliang_Liu_Resume.pdf` with the path to your PDF file.

#### API Response

The API will return a JSON object with the following structure:

```json
{
  "extractions": {...},
  "evaluations": {...},
  "final_rating": "..."
}
```

- `extractions`: Contains the extracted information for each of the eight O-1A criteria.
- `evaluations`: Contains the evaluation results for each criterion.
- `final_rating`: The overall qualification rating ([Low], [Medium], or [High]).

You can parse this JSON response to get the `final_rating` or any other specific information you need.

### Note

Regardless of the method you choose, the system processes the CV in the same way, applying the two-phase extraction and evaluation process to determine the O-1A visa qualification likelihood.
