#%% [Cell 1: Load CSV Files]
import pandas as pd
import glob
import os

data_path = 'data'
csv_files = glob.glob(os.path.join(data_path, '*.csv'))
print('CSV files found:', csv_files)
# %%

#%% [Cell 2: Combine CSV Files]
df_list = []

for file in csv_files:
    df = pd.read_csv(file)
    
    filename = os.path.basename(file).lower()
    if 'ddos' in filename:
        file_attack_type = 'DDoS'
    elif 'portscan' in filename:
        file_attack_type = 'PortScan'
    elif 'infilteration' in filename:
        file_attack_type = 'Infilteration'
    elif 'webattacks' in filename:
        file_attack_type = 'WebAttacks'
    elif 'pcap' in filename or 'iscx' in filename:
        file_attack_type = 'BENIGN'
    else:
        file_attack_type = 'Unknown'
        
    df['attack_type'] = file_attack_type
    df_list.append(df)
    
combined_df = pd.concat(df_list, ignore_index=True)

print(combined_df.head())
print('\nAttack type counts based on file source:')
print(combined_df['attack_type'].value_counts())
# %%

#%% [Cell 3: Save Combined DataFrame]
print(combined_df.head())
print(combined_df.info())
print(combined_df.describe())
# %%

#%% [Cell 4: Check For Missing Values]
print(combined_df.isnull().sum())
#%%

#%% [Cell 5: Check For Duplicates]
duplicates = combined_df.duplicated().sum()
print(f'Total duplicates: {duplicates}')
#%%

#%% [Cell 6: Look at Class Distribution]
import matplotlib.pyplot as plt
import seaborn as sns

sns.countplot(x='attack_type', data=combined_df)
plt.xticks(rotation=45)
plt.show()

print(combined_df['attack_type'].value_counts())
#%%

#%% [Cell 7: Remove Duplicates]
# Check the number of duplicates before removal
num_duplicates = combined_df.duplicated().sum()
print("Total duplicates before removal:", num_duplicates)

# Remove duplicate rows
combined_df = combined_df.drop_duplicates()

# Verify that there are no duplicates left
num_duplicates_after = combined_df.duplicated().sum()
print("Total duplicates after removal:", num_duplicates_after)

# %%

#%% [ Cell 8: Replace infinity values with NaN]
import numpy as np

numeric_cols = combined_df.select_dtypes(include=['int64', 'float64']).columns
# Check for infinity values in the numeric columns
combined_df.replace([np.inf, -np.inf], np.nan, inplace=True)
# Check for any remaining infinity values
print('Infinity values left?', np.isinf(combined_df[numeric_cols]).any().any())

# %% [Cell 9: Drop Missing Values]
# Drop rows with any missing values
combined_df.dropna(inplace=True)

# Check if there are any missing values left
print('Missing values left?', combined_df.isnull().sum().sum())

#%% [Cell 10: Feature Scaling]
from sklearn.preprocessing import StandardScaler

# Identify numeric columns in the DataFrame (int64 and float64 types)
numeric_cols = combined_df.select_dtypes(include=['int64', 'float64']).columns

# Initialize the StandardScaler
scaler = StandardScaler()

# Fit the scaler and transform the numeric features
combined_df[numeric_cols] = scaler.fit_transform(combined_df[numeric_cols])

# Review summary statistics to confirm the scaling effect
print(combined_df[numeric_cols].describe())
# %%

#%% [Cell 11: Splitting the Dataset]
from sklearn.model_selection import train_test_split

# Define features and target variable
X = combined_df.drop(['attack_type', ' Label'], axis=1)
y = combined_df['attack_type']

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

print(f'Training set shape: {X_train.shape}')
print(f'Testing set shape: {X_test.shape}')
# %%

#%% [Cell 12: Inspect X Data Types]
print(X.dtypes)
# %%

'''
Random Forest Model
'''

#%% [Cell 13: Model - Random Forest Setup]
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# Initialize the Random Forest classifier
rf_model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
# %%

#%% [Cell 14: Model - Train the Model]
rf_model.fit(X_train, y_train)
# %%

#%% [Cell 15: Predict with Random Forest]
y_pred_rf = rf_model.predict(X_test)
# %%    

#%% [Cell 16: Evaluate Random Forest Model]
print('Random Forest Classification Report:')
print(classification_report(y_test, y_pred_rf))

# Print the confusion matrix
print('Confusion Matrix:')
print(confusion_matrix(y_test, y_pred_rf))

# Print the accuracy score
print('Accuracy Score:', accuracy_score(y_test, y_pred_rf))
# %%

'''
END OF RANDOM FOREST MODEL
'''
'''
SVM Model
'''
#%% [Cell 17: Model - Sample the data]
from sklearn.model_selection import train_test_split

SAMPLE_SIZE = 30000

X_train_sample, _, y_train_sample, _ = train_test_split(
    X_train, y_train, 
    train_size=SAMPLE_SIZE, 
    stratify=y_train,
    random_state=42
)

# verifying the sample size
print(f'Sample training set shape: {X_train_sample.shape}')
print(f'Class distribution in sample:\n {y_train_sample.value_counts()}')
# %%

#%% [Cell 18: Tain SVM Model on Sampled Data]
from sklearn.svm import SVC

svm_model = SVC(kernel='rbf', class_weight='balanced', random_state=42)

# Train the SVM model on the sampled data
svm_model.fit(X_train_sample, y_train_sample)
# %%

# %% [Cell 19: Predict with SVM]
y_pred_svm = svm_model.predict(X_test)
# %%

#%% [Cell 20: Evaluate SVM Model]
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

print("SVM Classification Report:")
print(classification_report(y_test, y_pred_svm))

print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred_svm))

print("Accuracy Score:", accuracy_score(y_test, y_pred_svm))
# %%
'''
END OF SVM MODEL
'''
'''
DNN MODEL
'''
#%% [Cell 21: DNN Setup & Label Encoding]
import numpy as np
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical

le = LabelEncoder()
y_train_encoded = le.fit_transform(y_train)
y_test_encoded = le.transform(y_test)

num_classes = len(le.classes_)
y_train_categorical = to_categorical(y_train_encoded, num_classes=num_classes)
y_test_categorical = to_categorical(y_test_encoded, num_classes=num_classes)
# %%

#%% [Cell 22: DNN Model Setup]
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout

model = Sequential([
    Dense(128, activation='relu', input_shape=(X_train.shape[1],)),
    Dropout(0.2),
    Dense(64, activation='relu'),
    Dropout(0.2),
    Dense(num_classes, activation='softmax')
])

model.summary()
# %%

#%% [Cell 23: Compile & Train DNN Model]
from tensorflow.keras.callbacks import EarlyStopping

model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

history = model.fit(
    X_train, y_train_categorical,
    validation_split=0.2,
    epochs=30,
    batch_size=256,
    callbacks=[early_stop]
)

#%% [Cell 24: Predict with DNN]
y_pred_proba = model.predict(X_test)
y_pred_classes = np.argmax(y_pred_proba, axis=1)
# %%

#%% [Cell 25: Evaluate DNN Model]
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

print("DNN Classification Report:")
print(classification_report(y_test_encoded, y_pred_classes))

print("DNN Confusion Matrix:")
print(confusion_matrix(y_test_encoded, y_pred_classes))

print("DNN Accuracy Score:", accuracy_score(y_test_encoded, y_pred_classes))
# %% 

'''
END OF DNN MODEL
'''