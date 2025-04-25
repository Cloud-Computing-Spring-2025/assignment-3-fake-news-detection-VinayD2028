# Assignment-5-FakeNews-Detection
Build a simple machine learning pipeline using Spark MLlib to classify news articles as FAKE or REAL based on their content.
    
### Dataset
- Generated using 
```bash
    python3 Dataset_Generator.py
```
- File: `fake_news_sample.csv`

### 🧠 Brief Task Descriptions

| Task | Description |
|------|-------------|
| **Task 1: Load & Explore** | Load the dataset (`fake_news_sample.csv`) into a Spark DataFrame, display sample rows, count total articles, and extract distinct labels. |
| **Task 2: Text Preprocessing** | Clean and prepare the text by converting it to lowercase, tokenizing the text into words, and removing common stopwords. |
| **Task 3: Feature Extraction** | Transform the tokenized text into numerical features using TF-IDF. Convert categorical labels ("FAKE", "REAL") into numeric indices for model training. |
| **Task 4: Model Training** | Split the dataset into training and test sets. Train a logistic regression model using Spark MLlib and make predictions on the test set. |
| **Task 5: Model Evaluation** | Evaluate the model’s performance using Accuracy and F1 Score metrics. Save the results to a CSV file for reporting. |


### ▶️ How to Run the project
1. Install the faker module
    ```bash
        pip install faker
    ```
2. Install pyspark
    ```bash
        pip install pyspark
    ```
3. Run the news classifier script
    ```bash
        spark-submit news_classification_pipeline.py
    ```
