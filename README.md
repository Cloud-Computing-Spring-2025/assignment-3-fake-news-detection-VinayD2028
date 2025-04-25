# Assignment-5-FakeNews-Detection
Build a simple machine learning pipeline using Spark MLlib to classify news articles as FAKE or REAL based on their content.
    
### Dataset
- Generated using 
```bash
    python3 Dataset_Generator.py
```
- File: `fake_news_sample.csv`

### 🧪 Tasks and corresponding output files
1. Load and explore dataset (`task1_output.csv`)
2. Preprocess and tokenize text (`task2_output.csv`)
3. Extract TF-IDF features and encode labels (`task3_output.csv`)
4. Train logistic regression and predict (`task4_output.csv`)
5. Evaluate model (Accuracy, F1 Score in `task5_output.csv`)

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
