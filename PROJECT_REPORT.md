# Project Report

## Student Performance Predictor
### Using Decision Tree Classification in Python

---

**Subject:** Machine Learning / Data Science Mini Project
**Language:** Python 3
**Date:** March 2026

---

## Abstract

This project is about building a system that can predict a student's academic performance — whether it will be Good, Average, or Poor — using basic information like how much they study, how much they sleep, how often they attend class, and what they scored in their previous exam. I used a Decision Tree Classifier for this, trained on a dataset of 215 student records. The model ended up with about 86% accuracy, which shows that these everyday habits and metrics are genuinely useful for predicting how a student will perform. The final system was deployed as both a web app using Streamlit and a simple command-line tool.

---

## 1. Introduction

In most colleges and schools, we only find out a student is struggling after they've already failed an exam or missed too many classes. By that point it's often too late to help. The idea behind this project is that if we can predict performance early — based on data we already have — educators and students can take action before it becomes a problem.

Machine learning is well-suited for this kind of task. Given enough examples of student habits and their outcomes, a model can learn what patterns lead to good or poor results. I chose to keep the features simple and practical — things that are easy to track and don't require any complex data collection.

---

## 2. Problem Statement

Can we predict a student's academic performance category (Good, Average, or Poor) using only four basic behavioral features: daily study hours, nightly sleep hours, attendance percentage, and previous exam marks?

The secondary questions I wanted to explore were:
- Which of these features matters most?
- How accurately can a simple model make this prediction?
- Can we turn this into something a non-technical person can actually use?

---

## 3. Objectives

- Create a realistic, labeled dataset for student performance
- Build and evaluate a Decision Tree classification model
- Achieve reasonable accuracy (above 80%) on test data
- Deploy the model through a clean web interface
- Show data insights through visualizations
- Make the whole thing simple enough for anyone to run and understand

---

## 4. Methodology

### 4.1 Dataset

Since real student data is difficult to obtain due to privacy concerns, I generated a synthetic dataset of 215 records using Python. Each row represents one student with the following fields:

| Column           | Range         | Description                  |
|------------------|---------------|------------------------------|
| study_hours      | 1.0 – 10.0    | Daily average study hours    |
| sleep_hours      | 4.0 – 9.0     | Nightly sleep hours          |
| attendance       | 40% – 100%    | Class attendance percentage  |
| previous_marks   | 30 – 100      | Score in previous exam       |
| performance      | Good/Avg/Poor | Target label                 |

Labels were assigned using a scoring formula based on the relative importance of each feature:

```
score = (study_hours × 4) + (sleep_hours × 2) + (attendance × 0.3) + (previous_marks × 0.5)
```

If score ≥ 85 → Good, between 60 and 85 → Average, below 60 → Poor.

I gave study hours the highest weight because consistent effort is generally the strongest predictor of performance in an academic context.

### 4.2 Data Preprocessing

The target column (performance) is categorical, so I used scikit-learn's `LabelEncoder` to convert it to numbers before training. The mapping ends up as: Average=0, Good=1, Poor=2 (alphabetical order by default). No scaling was needed since Decision Trees don't rely on distance-based calculations.

### 4.3 Model Selection

I chose a **Decision Tree Classifier** because:
- It's easy to understand and explain — the model essentially learns a set of if-then rules
- It works well for classification problems with a small number of features
- It doesn't require feature scaling
- It's interpretable, which matters when making decisions about students

I set `max_depth=5` to prevent the tree from overfitting to the training data. Without a depth limit, decision trees tend to memorize the training set and perform poorly on new data.

### 4.4 Training and Evaluation

The dataset was split 80/20 into training and test sets, with stratification to ensure all three classes appear in both splits. I then evaluated using accuracy score and a full classification report showing precision, recall, and F1-score for each class.

### 4.5 Deployment

The trained model and label encoder were saved to disk using Python's `pickle` library. Two interfaces were built on top:

- **app.py** — A Streamlit web app with sliders for input, color-coded result boxes, improvement tips, and a charts section
- **main.py** — A simple colored terminal interface for users who prefer the command line

---

## 5. Tools and Technologies

| Tool / Library | Version | Purpose                                |
|----------------|---------|----------------------------------------|
| Python         | 3.8+    | Main programming language              |
| Pandas         | 2.0+    | Data loading and manipulation          |
| NumPy          | 1.24+   | Numerical computations                 |
| Scikit-learn   | 1.3+    | Decision Tree, LabelEncoder, metrics   |
| Matplotlib     | 3.7+    | Data visualization charts              |
| Streamlit      | 1.32+   | Web application interface              |
| Pickle         | built-in| Saving and loading the trained model   |

---

## 6. Results

### 6.1 Accuracy

The model achieved **86.05% accuracy** on the held-out test set of 43 students.

| Split    | Size |
|----------|------|
| Training | 172  |
| Testing  | 43   |

### 6.2 Classification Report

| Class   | Precision | Recall | F1-Score | Test Samples |
|---------|-----------|--------|----------|--------------|
| Average | 0.78      | 0.88   | 0.82     | 16           |
| Good    | 0.92      | 0.92   | 0.92     | 24           |
| Poor    | 1.00      | 0.33   | 0.50     | 3            |

The model performs best on the "Good" class, which makes sense since it has the most examples. The "Poor" class has limited samples (only 3 in the test set), which makes recall lower there — it's a known limitation when dealing with imbalanced classes.

### 6.3 Insights from the Charts

Looking at the visualizations generated by the project:

- Students who study 6+ hours daily are almost always in the "Good" category
- Attendance below 60% is a strong indicator of poor performance
- Sleep has a moderate effect — extreme values (under 5 hours or over 9) tend to correlate with worse performance
- Previous marks is the most consistent baseline predictor across all categories

---

## 7. Conclusion

This project successfully demonstrated that student performance can be predicted from simple behavioral inputs with around 86% accuracy. A Decision Tree is a good fit here — it's easy to explain, reasonably accurate, and doesn't require complex tuning.

The deployed system (web app + CLI) makes it accessible to non-technical users. The personalized tips based on input values also make the output more actionable, not just a label.

Main limitations: the dataset is synthetic, the "Poor" category is underrepresented, and the model only considers four features. That said, for a mini project the results are solid and the approach is sound.

---

## 8. Future Scope

There are several directions this project could be taken further:

1. **Real data** — Collecting anonymized data from actual institutions would make the model far more reliable and generalizable.

2. **More features** — Adding factors like socioeconomic background, parental education, access to resources, or mental health indicators could significantly improve accuracy.

3. **Stronger models** — Random Forest or XGBoost would likely outperform the Decision Tree, especially with a larger dataset.

4. **Fixing class imbalance** — Using SMOTE (Synthetic Minority Over-sampling Technique) to generate more "Poor" examples would help the model learn that class better.

5. **Continuous monitoring** — Instead of a one-time prediction, building a dashboard that tracks students week by week would be far more useful in practice.

6. **Explainability** — Adding SHAP values would let educators see exactly why the model predicted a particular outcome for a specific student.

7. **LMS integration** — Connecting to platforms like Moodle or Google Classroom could automate data collection and make predictions in real time.

---

## References

1. Quinlan, J.R. (1986). Induction of Decision Trees. *Machine Learning*, 1(1), 81–106.
2. Scikit-learn Documentation — https://scikit-learn.org/stable/
3. Streamlit Documentation — https://docs.streamlit.io
4. Romero, C., & Ventura, S. (2010). Educational Data Mining: A Review of the State of the Art. *IEEE Transactions on Systems, Man, and Cybernetics*, Part C.
5. Pandas Documentation — https://pandas.pydata.org/docs/

---

*End of Report*
