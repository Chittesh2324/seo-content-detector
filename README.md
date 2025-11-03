# SEO Content Quality & Duplicate Detector

The **SEO Content Quality & Duplicate Detector** is a Streamlit-based web application that analyzes and evaluates written or web content using advanced **Natural Language Processing (NLP)** and **Machine Learning (ML)** techniques.  
It helps identify **content quality issues**, detect **thin or low-value pages**, and find **duplicate content** across multiple URLs or text sources.

---

## Key Features

- **Content Quality Prediction** — Classifies text as High, Medium, or Low quality using a trained ML model.  
- **Text Metrics Analysis** — Calculates word count, sentence count, and Flesch Reading Ease (readability score).  
- **Machine Learning Integration** — Uses a Random Forest model trained on linguistic and readability features.  
- **Thin Content Detection** — Automatically flags pages with low word count (<300 words).  
- **Duplicate Content Detection** — Uses TF-IDF and cosine similarity to identify overlapping or identical pages.  
- **Interactive Visualizations**  
  - Confidence bar chart for ML prediction probabilities  
  - Readability gauge chart  
  - Content insight summary panel  
- **Downloadable Reports** — Export JSON results or duplicate reports directly from the interface.

---

## Project Structure

seo-content-detector/
├── data/
│ ├── data.csv
│ ├── extracted_content.csv
│ ├── features.csv
│ └── duplicates.csv
├── notebooks/
│ └── seo_pipeline.ipynb
├── models/
│ └── quality_model.pkl
├── streamlit_app/
│ ├── app.py
│ └── requirements.txt
├── .gitignore
└── README.md


---

## Installation and Setup

### 1. Clone the Repository
```bash
git clone https://github.com/yourusername/seo-content-detector.git
cd seo-content-detector/streamlit_app
2. Create a Virtual Environment
bash
Copy code
python -m venv venv
source venv/bin/activate       # On macOS/Linux
venv\Scripts\activate          # On Windows
3. Install Dependencies
bash
Copy code
pip install -r requirements.txt
4. Run the Streamlit Application
bash
Copy code
streamlit run app.py
5. Access the App
Open your browser and navigate to:

arduino
Copy code
http://localhost:8501
How It Works
Single URL Mode
Enter a webpage URL to fetch, clean, and analyze its content automatically.

The app extracts readable text, computes metrics, and predicts quality.

Results include metrics, ML predictions, readability analysis, and a downloadable JSON file.

Batch Mode
Upload a CSV file containing url and html_content columns.

The app detects duplicate content between pages and identifies thin pages.

You can download the complete duplicate analysis report as a CSV file.

Model Overview
The quality_model.pkl file is a Random Forest Classifier trained on textual and readability-based features extracted from SEO-optimized and non-optimized pages.
Core features used:

Word Count

Sentence Count

Flesch Reading Ease Score

The model predicts the overall content quality category.

Deployment Options
You can deploy this application using one of the following platforms:

Streamlit Cloud — ideal for public demos

Render or Railway.app — for quick and automated cloud hosting

AWS / Azure / GCP — suitable for enterprise deployment

Example deployment command:

bash
Copy code
streamlit run app.py --server.port 8501 --server.address 0.0.0.0
Example Use Cases
SEO analysts evaluating web page quality before publication

Content teams checking for duplicate or low-value content

Digital marketers tracking site-wide content quality

Data scientists exploring linguistic quality metrics

Requirements
Python 3.8 or higher

Required libraries (see requirements.txt):

streamlit

scikit-learn

beautifulsoup4

requests

pandas

numpy

plotly

textstat

joblib

License
This project is licensed under the MIT License.
You are free to use, modify, and distribute it with proper attribution.

Author
Developed by Data Science Candidate (2025)
For professional inquiries or collaborations, contact: your.email@example.com

yaml
Copy code

---

Would you like me to add a **"Demo Preview" section** with a placeholder for screenshots (e.g. `![App Screenshot](path/to/image.png)`) so it looks more professional on GitHub?