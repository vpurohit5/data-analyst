# Vaishnavi G. Purohit - Data Analyst Portfolio

## About Me
Data analyst with 3+ years of experience in data visualization, machine learning, and big data solutions. Proficient in tools such as Tableau, Power BI, and SAP HANA, with expertise in Python, SQL, and cloud platforms like AWS and Google Cloud. Skilled in extracting meaningful insights from large datasets to drive business strategy.

## Portfolio Projects
Below are selected projects that showcase my analytical skills, technical expertise, and business acumen.

1. [Predictive Analytics for Student Performance](#project-1-predictive-analytics-for-student-performance)
2. [Social Media Sentiment Analysis](#project-2-social-media-sentiment-analysis)
3. [Customer Segmentation for Marketing Campaigns](#project-3-customer-segmentation-for-marketing-campaigns)
4. [Real-Time Air Quality Monitoring Dashboard](#project-4-real-time-air-quality-monitoring-dashboard)
5. [Sales Analysis for Revenue Optimization](#project-5-sales-analysis-for-revenue-optimization)
6. [Supply Chain Data Analysis and Forecasting](#project-6-supply-chain-data-analysis-and-forecasting)


### Project 1: Predictive Analytics for Student Performance

- Skills Used: Python, Logistic Regression, Random Forests, Data Preprocessing
- Description: Developed a model to predict student performance, helping educators identify at-risk students early.
- Outcome: Improved early intervention by 20%, enhancing support for students likely to need academic assistance.


```python
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Load data
data = pd.read_csv('student_performance.csv')
X = data[['attendance', 'homework', 'test_scores']]
y = data['performance']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Evaluate model
predictions = model.predict(X_test)
accuracy = accuracy_score(y_test, predictions)
print(f"Accuracy: {accuracy}")

