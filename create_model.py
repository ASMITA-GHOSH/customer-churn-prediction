import pandas as pd
import pickle
from sklearn.ensemble import RandomForestClassifier
import numpy as np

print("🔄 Creating training data...")

# Create training data
np.random.seed(42)
n_samples = 5000

data = {
    'tenure': np.random.randint(1, 73, n_samples),
    'MonthlyCharges': np.random.uniform(20, 120, n_samples),
    'TotalCharges': np.random.uniform(100, 9000, n_samples),
    'Contract': np.random.choice([0, 1, 2], n_samples, p=[0.4, 0.35, 0.25]),
    'InternetService': np.random.choice([0, 1, 2], n_samples, p=[0.3, 0.4, 0.3]),
}

df = pd.DataFrame(data)

# Create churn pattern
df['Churn'] = (
    (df['tenure'] < 12) * 0.4 +
    (df['MonthlyCharges'] > 70) * 0.3 +
    (df['Contract'] == 0) * 0.2 +
    np.random.random(n_samples) * 0.1
)
df['Churn'] = (df['Churn'] > 0.5).astype(int)

print(f"✅ Created {n_samples} records")
print(f"📊 Churn rate: {df['Churn'].mean():.1%}")

# Features
feature_columns = ['tenure', 'MonthlyCharges', 'TotalCharges', 'Contract', 'InternetService']
X = df[feature_columns]
y = df['Churn']

# Train model
print("🤖 Training model...")
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X, y)

accuracy = model.score(X, y)
print(f"✅ Model accuracy: {accuracy:.2%}")

# Save model
print("💾 Saving model...")
with open('churn_model.pkl', 'wb') as f:
    pickle.dump(model, f)

with open('features.pkl', 'wb') as f:
    pickle.dump(feature_columns, f)

print("✅ churn_model.pkl created!")
print("✅ features.pkl created!")
print("\n🎉 Training complete! Now run: streamlit run app.py")