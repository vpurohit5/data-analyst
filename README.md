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

#### Code Example:

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
```

### Project 2: Social Media Sentiment Analysis

- Skills Used: Natural Language Processing (NLP), Python, TextBlob, Tableau
- Description: Analyzed Twitter and LinkedIn data to assess public sentiment toward major events.
- Outcome: Insights were utilized for strategic communications and brand positioning.

#### Code :

```python
import pandas as pd
from textblob import TextBlob

# Load social media data
data = pd.read_csv('social_media_posts.csv')
data['sentiment'] = data['text'].apply(lambda x: TextBlob(x).sentiment.polarity)

# Simple sentiment analysis
data['sentiment_label'] = data['sentiment'].apply(lambda x: 'positive' if x > 0 else 'negative' if x < 0 else 'neutral')

# Plot results with Tableau
data.to_csv('sentiment_analysis_output.csv')
```

### Project 3: Customer Segmentation for Marketing Campaigns

- Skills Used: Clustering, Python, Scikit-Learn, K-Means, Data Visualization
- Description: Utilized clustering to segment customers based on purchasing behaviors, improving targeted marketing.
- Outcome: Achieved a 30% increase in marketing ROI by enabling more personalized campaign strategies.

#### Code:

```python
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns

# Load customer data
data = pd.read_csv('customer_data.csv')
X = data[['annual_income', 'spending_score']]

# KMeans clustering
kmeans = KMeans(n_clusters=3, random_state=0).fit(X)
data['cluster'] = kmeans.labels_

# Visualization
sns.scatterplot(x='annual_income', y='spending_score', hue='cluster', data=data)
plt.title('Customer Segmentation')
plt.show()
```

### Project 4: Real-Time Air Quality Monitoring Dashboard

- Skills Used: Tableau, Python, Data Cleaning, Real-Time Data Processing
- Description: Built a real-time dashboard for monitoring air quality across various locations.
- Outcome: Improved response times for air quality alerts by 20%, aiding environmental research efforts.

#### Code :

```python
# Preprocess data in Python
import pandas as pd

data = pd.read_csv('air_quality_data.csv')
data['datetime'] = pd.to_datetime(data['timestamp'])
data.set_index('datetime', inplace=True)

# Data cleaned and ready for real-time visualization in Tableau
data.to_csv('cleaned_air_quality_data.csv')
```

### Project 5: Sales Analysis for Revenue Optimization

- Skills Used: SQL, Excel, Power BI, Data Visualization, Tableau
- Description: Analyzed historical sales data to identify trends, seasonal patterns, and key revenue drivers.
- Outcome: Generated actionable insights that helped optimize pricing and promotional strategies.

#### Code (SQL):

```sql
SELECT
    product_category,
    MONTH(sale_date) AS month,
    SUM(sales_amount) AS total_sales
FROM
    sales_data
GROUP BY
    product_category, month
ORDER BY
    total_sales DESC;
```


### Project 6: Supply Chain Data Analysis and Forecasting

- Skills Used: Time Series Analysis, Python, ARIMA Model, Data Forecasting
- Description: Conducted supply chain forecasting using time-series data to optimize inventory levels.
- Outcome: Improved inventory management by 15%, reducing stockouts and excess holding costs.

#### Code:

```python
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA

# Load time series data
data = pd.read_csv('inventory_data.csv')
data['date'] = pd.to_datetime(data['date'])
data.set_index('date', inplace=True)

# ARIMA model for forecasting
model = ARIMA(data['inventory_level'], order=(5, 1, 0))
model_fit = model.fit()
forecast = model_fit.forecast(steps=12)
print(forecast)
```


### Additional Skills and Tools
- Languages: Python, SQL, C++, C#
- Data Visualization: Tableau, Power BI, Matplotlib, Seaborn
- Machine Learning: Scikit-Learn, Deep Learning with Keras
- Big Data: SAP HANA, Hadoop, Spark

---

### Contact
- LinkedIn: [Vaishnavi Purohit](https://www.linkedin.com/in/vaishnavi-purohit-824664250)
- Email: vaishnavipurohitt81@gmail.com


