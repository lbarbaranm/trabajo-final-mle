import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

def train_model(data):
    X = data.drop('target', axis=1)
    y = data['target']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = LinearRegression()
    model.fit(X_train, y_train)
    return model

if __name__ == "__main__":
    data = pd.read_csv('data/processed/clean_data.csv')
    trained_model = train_model(data)
    trained_model.save('models/trained_model.pkl')