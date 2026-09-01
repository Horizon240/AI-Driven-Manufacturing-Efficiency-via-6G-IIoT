import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score

# Load Data
df = pd.read_csv(r"C:\Users\kshit\Downloads\Thales_Group_Manufacturing.csv")

# Preprocessing
le_mode = LabelEncoder()
le_target = LabelEncoder()
df['Operation_Mode'] = le_mode.fit_transform(df['Operation_Mode'])
df['Efficiency_Status'] = le_target.fit_transform(df['Efficiency_Status'])

# Feature Engineering
epsilon = 1e-6 # Prevent division by zero
df['Energy_Efficiency_Ratio'] = df['Production_Speed_units_per_hr'] / (df['Power_Consumption_kW'] + epsilon)
df['Error_to_Output_Ratio'] = df['Error_Rate_%'] / (df['Production_Speed_units_per_hr'] + epsilon)
df['Network_Reliability_Score'] = df['Network_Latency_ms'] * (1 + df['Packet_Loss_%'] / 100)
df['Sensor_Stress_Index'] = df['Temperature_C'] * df['Vibration_Hz']

features = [
    'Operation_Mode', 'Temperature_C', 'Vibration_Hz', 'Power_Consumption_kW', 
    'Network_Latency_ms', 'Packet_Loss_%', 'Quality_Control_Defect_Rate_%', 
    'Production_Speed_units_per_hr', 'Predictive_Maintenance_Score', 'Error_Rate_%',
    'Energy_Efficiency_Ratio', 'Error_to_Output_Ratio', 'Network_Reliability_Score', 'Sensor_Stress_Index'
]

X = df[features]
y = df['Efficiency_Status']

# Train/Test Split & Scaling
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Baseline Logistic Regression 
lr_model = LogisticRegression(solver='lbfgs', max_iter=1000)
lr_model.fit(X_train_scaled, y_train)

# Evaluation
y_pred = lr_model.predict(X_test_scaled)
print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print(classification_report(y_test, y_pred, target_names=le_target.classes_))


from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier

# Handle Class Imbalance with SMOTE
print("\n--- Advanced Model: XGBoost with SMOTE ---")
smote = SMOTE(random_state=42)
X_train_smote, y_train_smote = smote.fit_resample(X_train_scaled, y_train)

# Train XGBoost Classifier
xgb_model = XGBClassifier(
    objective='multi:softprob',
    num_class=3,
    eval_metric='mlogloss',
    use_label_encoder=False,
    random_state=42,
    max_depth=6,
    learning_rate=0.1,
    n_estimators=200
)

xgb_model.fit(X_train_smote, y_train_smote)

# Evaluation
y_pred_xgb = xgb_model.predict(X_test_scaled)
xgb_acc = accuracy_score(y_test, y_pred_xgb)

print(f"XGBoost Accuracy: {xgb_acc:.4f}")
print("\nXGBoost Classification Report:")
print(classification_report(y_test, y_pred_xgb, target_names=le_target.classes_))

#Saving
import joblib
joblib.dump(xgb_model, 'xgb_efficiency_model.pkl')
joblib.dump(scaler, 'feature_scaler.pkl')
joblib.dump(le_target, 'target_encoder.pkl')