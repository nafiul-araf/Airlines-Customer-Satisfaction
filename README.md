# Airlines Customer Satisfaction Analysis using Machine Learning

**Click on the Image Below to Go to the App 👇**

[![image](https://github.com/user-attachments/assets/974d69bd-9a88-4089-bff9-6a99fcb88ecc)](https://abc-airlines-customer-satisfaction.streamlit.app/)


### **Project Summary: Airlines Customer Satisfaction Analysis**

#### **Project Statement**  
The aim of this project was to analyze and predict customer satisfaction for an airline based on various features such as demographics, travel preferences, and service quality metrics. By leveraging machine learning techniques, the goal was to create a reliable model capable of identifying factors contributing to customer satisfaction and predicting customer sentiment with high accuracy.

---

#### **Purpose**  
1. **Customer Insights**: Understand the drivers behind customer satisfaction and dissatisfaction to guide strategic improvements.  
2. **Business Value**: Provide actionable insights to enhance customer retention, improve services, and boost overall profitability.  
3. **Predictive Modeling**: Build a robust machine learning model to predict customer satisfaction effectively, enabling data-driven decision-making.

---

#### **Process**  
1. **Data Collection and Preprocessing**  
   - The dataset was cleaned, missing values handled, and categorical variables encoded.  
   - Exploratory data analysis (EDA) identified key trends, patterns, and correlations in the dataset.

2. **Feature Engineering**  
   - Relevant features were selected using techniques like VIF analysis to reduce multicollinearity.  
   - Numerical and categorical variables were transformed appropriately for modeling.

3. **Model Development and Evaluation**  
   - Multiple models, including Logistic Regression, Random Forest, and LightGBM, were trained.  
   - Optuna was employed for hyperparameter tuning, maximizing model performance.  
   - Performance was evaluated using metrics such as accuracy, precision, recall, F1-score, and AUC.

4. **Findings**  
   - LightGBM emerged as the best-performing model with:  
     - Training accuracy: 91.91%  
     - Test accuracy: 86.16%  
     - AUC: 0.94  
   - This model demonstrated excellent generalization and predictive capabilities, making it the ideal choice for deployment.

5. **Deployment**  
   - The final LightGBM model was deployed using **Streamlit Cloud**, providing an interactive interface for predictions.  
   - Users can input customer data and instantly receive satisfaction predictions, facilitating real-time decision-making.

---

#### **Findings**  
1. **Key Insights**:  
   - Service quality and travel experience are primary drivers of customer satisfaction.  
   - Demographic factors also contribute but to a lesser extent compared to service-related attributes.

2. **Model Performance**:  
   - LightGBM outperformed other models in terms of accuracy and AUC, indicating its suitability for real-world applications.  
   - The classification report revealed balanced performance across all metrics, ensuring reliable predictions.

---

#### **Deployment and Impact**  
The deployment on **Streamlit Cloud** enables stakeholders to leverage the model conveniently and effectively. By integrating predictive capabilities into the decision-making process, the airline can:  
- Identify dissatisfied customers proactively.
- Tailor services to improve satisfaction rates.  
- Enhance customer retention and boost loyalty, translating into better business outcomes.  

This project demonstrates the power of machine learning in addressing real-world business challenges and creating data-driven solutions for customer-centric industries.

---

## **Project Structure**

```
project-root/
│
├── model/
│   ├── model_data.pkl                # Serialized machine learning model and preprocessing data
│
├── ABC_Airlines.JPG                 # Project logo or related image
├── readme.md                        # Documentation file
├── main.py                          # Streamlit application file
├── requirements.txt                 # List of required Python packages
├── utils.py                         # Utility functions for prediction and preprocessing
```

---

## **Prerequisites**

Before running the project, ensure you have the following installed on your system:
- **Python** (version 3.8 or above)
- **pip** (Python package installer)

---

## **Setup Instructions**

Follow these steps to set up and run the project on your local machine:

### **1. Clone the Repository**

Use the following command to clone the repository:

```bash
git clone https://github.com/<your-username>/airlines-customer-satisfaction.git
```

Navigate to the project directory:

```bash
cd airlines-customer-satisfaction/root
```

### **2. Create a Virtual Environment**

It is recommended to create a virtual environment to isolate the project dependencies. Run the following commands:

```bash
python -m venv env
source env/bin/activate        # For Linux/Mac
env\Scripts\activate           # For Windows
```

### **3. Install Dependencies**

Install the required Python packages using the `requirements.txt` file:

```bash
pip install -r requirements.txt
```

### **4. Run the Streamlit Application**

Launch the Streamlit app with the following command:

```bash
streamlit run main.py
```

---

## **Usage**

1. After running the Streamlit app, it will open in your default web browser.
2. Upload customer data or enter details manually into the input fields.
3. Click the **"Predict"** button to see the prediction result (Satisfied/Dissatisfied).
4. Review insights and recommendations based on the prediction.

---

## **Files and Directories**

- **`model/model_data.pkl`**  
  Contains the serialized LightGBM model and preprocessing pipeline.

- **`main.py`**  
  The main Streamlit application file to launch the web interface.

- **`requirements.txt`**  
  A list of Python libraries required to run the project.

- **`utils.py`**  
  Contains helper functions for preprocessing and prediction.

- **`ABC_Airlines.JPG`**  
  A visual asset for branding and representation in the project.

---
