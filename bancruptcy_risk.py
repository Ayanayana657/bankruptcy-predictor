import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

df = pd.read_csv('fin_buncruptcy.csv')

X = df.drop('Bankruptcy', axis=1)
y = df['Bankruptcy']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print('Точность модели:', accuracy)

new_data = pd.DataFrame({
    'Liquidity': [1.2],
    'Leverage': [0.9],
    'Profitability': [0.15]
})

prediction = model.predict(new_data)
print('Bankruptcy:', 'Yes' if prediction[0] == 1 else 'No')



