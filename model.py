import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

# Dataset
data = {
    "hours": [1,2,3,4,5,6,7,8],
    "attendance": [50,60,65,70,75,80,85,90],
    "marks": [30,35,40,50,55,65,70,80],
    "result": [0,0,0,1,1,1,1,1]
}

df = pd.DataFrame(data)

# Features & Target
X = df[["hours", "attendance", "marks"]]
y = df["result"]

# Train model
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
model = LogisticRegression()
model.fit(X_train, y_train)

# Prediction
prediction = model.predict([[5, 80, 60]])

print("Prediction:", "Pass" if prediction[0] == 1 else "Fail")
