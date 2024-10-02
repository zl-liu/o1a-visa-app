# O-1A Visa Application Evaluator

This project is an automated tool that evaluates an applicant's likelihood of success for an O-1A visa based on their CV. The application processes a resume/CV PDF file, assesses the evidence based on O-1A criteria, and outputs a final rating of [Low], [Medium], or [High].

## Screenshots

![O-1A Visa Application Evaluator Screenshot1](app_screenshot1.png)
![O-1A Visa Application Evaluator Screenshot4](app_screenshot4.png)

## Installation

1. Clone the app repository:
   ```bash
   git clone https://github.com/zl-liu/o1a-visa-app.git
   cd o1a-visa-app
   ```

2. Create the Conda environment from the environment.yml file:
   ```bash
   conda env create -f environment.yml
   ```

3. Activate the Conda environment:
   ```bash
   conda activate myenv
   ```

## Starting the Application

1. Run the FastAPI server:
   In the first terminal, run the following command to start the FastAPI server:
   ```bash
   uvicorn main:app --host 0.0.0.0 --port 8000 --reload
   ```
   This will launch the backend server, which processes the uploaded PDF files.

2. Run the Chainlit frontend:
   In a separate terminal, start the Chainlit frontend:
   ```bash
   chainlit run my_cl_app.py --host 0.0.0.0 --port 8500
   ```
   This will launch the web interface for uploading the resume/CV PDF file and receiving the assessment.

## Usage

1. Open your browser and go to the Chainlit app at:
   ```
   http://0.0.0.0:8500
   ```

2. Upload a resume/CV PDF file when prompted.

3. Wait for the system to process the file. After processing, the system will output the likelihood of success for the O-1A visa application as either [Low], [Medium], or [High].

## License

This project is licensed under the MIT License.
