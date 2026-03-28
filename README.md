# Student Performance Predictor

This is a machine learning project I built that predicts whether a student will perform **Good**, **Average**, or **Poor** based on four simple inputs — study hours, sleep hours, attendance, and previous marks.

The idea behind this project is that we can use basic day-to-day data about a student to get an early sense of how they might perform, so teachers or students themselves can take action before it's too late.

---

## Project Files

```
student_performance/
├── data.csv              ← dataset with 215 student records
├── model.py              ← trains the ML model and handles predictions
├── app.py                ← Streamlit web app (the main UI)
├── main.py               ← CLI version if you prefer the terminal
├── requirements.txt      ← all required Python libraries
├── README.md             ← you're reading this right now
└── visualizations.png    ← gets created automatically after training
```

---

## How to Set Up

### 1. Download or clone the project
```bash
git clone <your-repo-url>
cd student_performance
```

### 2. (Optional but recommended) Create a virtual environment
```bash
python -m venv venv
source venv/bin/activate       # on Mac/Linux
venv\Scripts\activate          # on Windows
```

### 3. Install the required libraries
```bash
pip install -r requirements.txt
```

---

## How to Run

### Web App (recommended)
```bash
streamlit run app.py
```
This will open the app in your browser at `http://localhost:8501`. You can adjust sliders for each feature and click predict.

### Command Line
```bash
python main.py
```
Simple terminal interface where you type in values and get a result.

### Train model manually
```bash
python model.py
```
This trains the model from scratch and saves everything to disk.

---

## Input Features

| Feature          | What it means                              |
|------------------|--------------------------------------------|
| `study_hours`    | How many hours per day the student studies |
| `sleep_hours`    | How many hours of sleep per night          |
| `attendance`     | Percentage of classes attended             |
| `previous_marks` | Score in the last exam (out of 100)        |

**Output:** `Good`, `Average`, or `Poor`

---

## About the Model

I used a **Decision Tree Classifier** from scikit-learn. It's a simple but effective algorithm that works by learning a set of if-else rules from the training data. I set `max_depth=5` to avoid overfitting.

- Training/Testing split: 80% / 20%
- Encoding: LabelEncoder for the target variable
- Final accuracy: around **86%**

---

## Libraries Used

- Python 3.8+
- pandas — for reading and working with the data
- scikit-learn — for the model, encoding, and evaluation
- matplotlib — for generating the charts
- streamlit — for the web interface

---

## Sample Output

```
Study Hours:    7.0
Sleep Hours:    7.5
Attendance:     85%
Previous Marks: 74

→ Predicted Performance: Good ✅
```
