import kagglehub

# Download latest version
path = kagglehub.dataset_download("blastchar/telco-customer-churn")

# #print("Path to dataset files:", path)

# #print(path)

import os

path = r"C:\Users\Avinash\.cache\kagglehub\datasets\blastchar\telco-customer-churn\versions\1"

# #print(os.listdir(path))




import pandas as pd

file_path = r"C:\Users\Avinash\.cache\kagglehub\datasets\blastchar\telco-customer-churn\versions\1\WA_Fn-UseC_-Telco-Customer-Churn.csv"

df = pd.read_csv(file_path)

# #print(df.head())
#print(df.info())
#print(df.describe())
#print(df.isnull().sum())
#print(df.columns)



#print(df.head())


df.drop("customerID", axis=1, inplace=True)
#print(df.columns)

#print(df['Churn'].value_counts())


# print(
#     df["TotalCharges"].dtype
# )

df["TotalCharges"] = pd.to_numeric(
    df["TotalCharges"],
    errors="coerce"
)

#print(df["TotalCharges"].isnull().sum())

# #print(df['TotalCharges'].isnull())
#print(df['TotalCharges'].isna().sum())


# Coverting Null Values into NaN

df["TotalCharges"] = pd.to_numeric(
    df["TotalCharges"],
    errors="coerce"
)

#print(df['TotalCharges'].isnull().sum())

# Now Filling that empty place by zero
df["TotalCharges"].fillna(
    df["TotalCharges"].median(),
    inplace=True
)

#print(df['TotalCharges'].isnull().sum())

# Checking target distribution
#print(df["Churn"].value_counts())


import seaborn as sns
import matplotlib.pyplot as plt

# sns.countplot(x="Churn", data=df)

# plt.show()


# Analyze categorical features
"Contract vs Churn"
sns.countplot(
    x="Contract",
    hue="Churn",
    data=df
)

# plt.xticks(rotation=45)
# plt.show()

"Internet Service vs Churn"
sns.countplot(
    x="InternetService",
    hue="Churn",
    data=df
)

# plt.show()


"Payment Method vs Churn"
sns.countplot(
    x="PaymentMethod",
    hue="Churn",
    data=df
)

# plt.xticks(rotation=45)
# plt.show()

"Tenure vs Churn"
sns.boxplot(
    x="Churn",
    y="tenure",
    data=df
)

# plt.show()


"Monthly Charges vs Churn"
sns.boxplot(
    x="Churn",
    y="MonthlyCharges",
    data=df
)

# plt.show()


#print(df.info())



# Seperate Feature for model
X = df.drop("Churn",axis=1)
y = df["Churn"]


#print(X,y)

# Converting y into numeric

"using label encoder"

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
y_encoded = le.fit_transform(y)
#print(y_encoded)

X = pd.get_dummies(X, drop_first=True)
#print(X)



"Making Function to save time"

from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

def model_evaluation(y_test, y_pred):
    
    accuracy = accuracy_score(y_test, y_pred)
    
    #print("Accuracy:", accuracy)
    
    #print("\nConfusion Matrix:")
    #print(confusion_matrix(y_test, y_pred))
    
    #print("\nClassification Report:")
    #print(classification_report(y_test, y_pred))




# Now creating model
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

#print(X_train.shape)
#print(X_test.shape)
#print(y_train.shape)
#print(y_test.shape)


from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


"Model Training"

# Model1 Logistic Regression
from sklearn.linear_model import LogisticRegression

model_lr = LogisticRegression(max_iter=1000)

model_lr.fit(X_train_scaled, y_train)   

y_pred_lr = model_lr.predict(X_test_scaled)   

accuracy_lr = accuracy_score(y_test, y_pred_lr)

#print(f'Accuracy of Logistic Regression is : {accuracy_lr}')

"Confusion Matrix"
from sklearn.metrics import confusion_matrix

cm_lr = confusion_matrix(
    y_test,
    y_pred_lr
)

#print(cm_lr)


"Full Report of Classification"
from sklearn.metrics import classification_report

# print(
#     classification_report(
#         y_test,
#         y_pred_lr
#     )
# )

# Model 2 Random Forest
from sklearn.ensemble import RandomForestClassifier

model_rf = RandomForestClassifier()
model_rf.fit(X_train,y_train)
y_pred_rf = model_rf.predict(X_test)
#print(y_pred_rf)
#print(y_test)

accuracy_rf = accuracy_score(y_test,y_pred_rf)
#print(f'Accuracy of Random Forest  is : {accuracy_rf}')

cm_rf = confusion_matrix(y_test,y_pred_rf)
#print(f"Confusion Matrix for Random Forest is : \n{cm_rf}")

classification_report_rf = classification_report(y_test,y_pred_rf)
#print(f"Classification Report for Random Forest is : \n{classification_report_rf}")


# Model 3 KNN
from sklearn.neighbors import KNeighborsClassifier
model_knn = KNeighborsClassifier()
#print(f"Model Name : {model_knn}")
#print(f'By Default model parameter are : \n{model_knn.get_params()}')
model_knn.fit(X_train_scaled,y_train)
y_pred_knn = model_knn.predict(X_test_scaled)
#print("Predicted Value",y_pred_knn)
#print(y_test)


accuracy_knn = accuracy_score(y_test,y_pred_knn)
#print(f'Accuracy of KNN is : {accuracy_knn}')

cm_knn = confusion_matrix(y_test,y_pred_knn)
#print(f"Confusion Matrix for KNN is : \n{cm_knn}")

classification_report_knn = classification_report(y_test,y_pred_knn)
#print(f"Classification Report for KNN is : \n{classification_report_knn}")


# To SAVE TIME WE ARE USING FUNCTION
model_detail_knn = model_evaluation(y_test,y_pred_knn)



# Model 4 Naive Bayes
from sklearn.naive_bayes import GaussianNB
model_nb = GaussianNB()
model_nb.fit(X_train,y_train)
y_pred_nb = model_nb.predict(X_test)
#print(y_pred_nb)
#print(y_test)

accuracy_nb = accuracy_score(y_test,y_pred_nb)
#print(f'Accuracy of NB is : {accuracy_nb}')

cm_nb = confusion_matrix(y_test,y_pred_nb)
#print(f"Confusion Matrix for NB is : \n{cm_nb}")

classification_report_nb = classification_report(y_test,y_pred_nb)
#print(f"Classification Report for NB is : \n{classification_report_nb}")

model_detail_nb = model_evaluation(y_test,y_pred_nb)



# Model 5 Decision Tree
from sklearn.tree import DecisionTreeClassifier
model_dt = DecisionTreeClassifier()
#print(model_dt.get_params())
model_dt.fit(X_train_scaled,y_train)
y_pred_dt = model_dt.predict(X_test_scaled)
#print(y_pred_dt)
#print(y_test)

accuracy_dt = accuracy_score(y_test,y_pred_dt)
#print(f'Accuracy of DT is : {accuracy_dt}')

cm_dt = confusion_matrix(y_test,y_pred_dt)
#print(f"Confusion Matrix for DT is : \n{cm_dt}")

classification_report_dt = classification_report(y_test,y_pred_dt)
#print(f"Classification Report for DT is : \n{classification_report_dt}")


model_detail_dt = model_evaluation(y_test,y_pred_dt)
#print(model_detail_dt)



# Model 6 SVM
from sklearn.svm import SVC
model_svm = SVC()
#print(model_svm.get_params())
model_svm.fit(X_train_scaled,y_train)
y_pred_svm = model_svm.predict(X_test_scaled)
#print(y_pred_svm)
#print(y_test)

accuracy_svm = accuracy_score(y_test,y_pred_svm)
#print(f'Accuracy of SVM is : {accuracy_svm}')

cm_svm = confusion_matrix(y_test,y_pred_svm)
#print(f"Confusion Matrix for SVM is : \n{cm_svm}")

classification_report_svm = classification_report(y_test,y_pred_svm)
#print(f"Classification Report for SVM is : \n{classification_report_svm}")


model_detail_svm = model_evaluation(y_test,y_pred_svm)
#print(model_detail_svm)



#print(df.head())
#print(df.columns)




"Making an Dictionary for comparision"
Dis_acc = {
 "Logistic Regression" : accuracy_lr,
 "Random Forest" : accuracy_rf,
 "KNN" : accuracy_knn,
 "Naive Bayes" : accuracy_nb,
 "Decision Tree" : accuracy_dt,
 "SVM" : accuracy_svm
}

# Creating Table of dictionary
import pandas as pd

# converting dictionary into dataframe
model_comparision = pd.DataFrame(
    list(Dis_acc.items()),
    columns=["Model","Accuracy"]
)

# Sort by accuracy (highest first)
model_comparision_sort = model_comparision.sort_values(
    by="Accuracy",
    ascending=False
)

#print(f'Table made by Dictionary : \n{model_comparision}')
#print(f"Sorted Dictionary is : \n{model_comparision_sort}")



#print(model_lr) 

""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
# ================================
# STEP 2: MODEL TUNING
# ================================
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""


"""Model Hyper Tunning"""

"""Used GridSearchcv for Hyprer Tunning"""

model_params = {

    "Logistic Regression": {
        "model": LogisticRegression(max_iter=1000),
        "params": {
            "C": [0.01, 0.1, 1, 10, 100],
            "solver": ["liblinear", "lbfgs"],
            "penalty": ["l2"]
        }
    },


    "Random Forest": {
        "model": RandomForestClassifier(),
        "params": {
            "n_estimators": [100, 200, 300],
            "max_depth": [5, 10, 20, None],
            "min_samples_split": [2, 5, 10],
            "min_samples_leaf": [1, 2, 4]
        }
    },


    "KNN": {
        "model": KNeighborsClassifier(),
        "params": {
            "n_neighbors": [3, 5, 7, 9, 11],
            "weights": ["uniform", "distance"],
            "metric": ["euclidean", "manhattan"]
        }
    },


    "Decision Tree": {
        "model": DecisionTreeClassifier(),
        "params": {
            "criterion": ["gini", "entropy"],
            "max_depth": [5, 10, 20, None],
            "min_samples_split": [2, 5, 10],
            "min_samples_leaf": [1, 2, 4]
        }
    },


    "SVM": {
        "model": SVC(),
        "params": {
            "C": [0.1, 1, 10, 100],
            "kernel": ["linear", "rbf"],
            "gamma": ["scale", "auto"]
        }
    },


    "Naive Bayes": {
        "model": GaussianNB(),
        "params": {
            "var_smoothing": [1e-9, 1e-8, 1e-7, 1e-6]
        }
    }

}


# from sklearn.model_selection import GridSearchCV

# results = []

# for model_name, mp in model_params.items():

#     print(f"Training {model_name}...")

#     grid = GridSearchCV(
#         mp["model"],
#         mp["params"],
#         cv=5,
#         scoring="accuracy",
#         n_jobs=-1
#     )

#     grid.fit(X_train_scaled, y_train)

#     results.append({
#         "Model": model_name,
#         "Best Accuracy": grid.best_score_,
#         "Best Parameters": grid.best_params_
#     })


# results_df = pd.DataFrame(results)

# results_df = results_df.sort_values(
#     by="Best Accuracy",
#     ascending=False
# )

# print(results_df)


# # Convert result into dataframe
# grid_results_df = pd.DataFrame(results)


# # Sort result
# grid_results_df = grid_results_df.sort_values(
#     by="Best Accuracy",
#     ascending=False
# )


# # Show result
# print(f"This is the saved data in .csv format : {grid_results_df}")


# # Save result
# grid_results_df.to_csv(
#     "grid_search_results.csv",
#     index=False
# )






"""Now using RandomSearchCV for Hyperparameter Tuning"""

from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB

import pandas as pd


random_params = {

    "Logistic Regression": {
        "model": LogisticRegression(max_iter=2000),
        "params": {
            "C": [0.001, 0.01, 0.1, 1, 10, 100],
            "solver": ["lbfgs", "liblinear"],
        }
    },


    "Random Forest": {
        "model": RandomForestClassifier(random_state=42),
        "params": {
            "n_estimators": [100, 200, 300],
            "max_depth": [5, 10, 20, None],
            "min_samples_split": [2, 5, 10],
            "min_samples_leaf": [1, 2, 4]
        }
    },


    "KNN": {
        "model": KNeighborsClassifier(),
        "params": {
            "n_neighbors": [3,5,7,9,11,15],
            "weights": ["uniform", "distance"],
            "metric": ["euclidean", "manhattan"]
        }
    },


    "Decision Tree": {
        "model": DecisionTreeClassifier(random_state=42),
        "params": {
            "criterion": ["gini", "entropy"],
            "max_depth": [5,10,20,None],
            "min_samples_split": [2,5,10],
            "min_samples_leaf": [1,2,4]
        }
    },


    "SVM": {
        "model": SVC(),
        "params": {
            "C": [0.01,0.1,1,10],
            "kernel": ["linear","rbf"],
            "gamma": ["scale","auto"]
        }
    },


    "Naive Bayes": {
        "model": GaussianNB(),
        "params": {
            "var_smoothing": [1e-9,1e-8,1e-7,1e-6]
        }
    }

}



# results_random = []


# for model_name, mp in random_params.items():

#     print(f"\nTraining {model_name}...")


#     random_search = RandomizedSearchCV(
#         estimator=mp["model"],
#         param_distributions=mp["params"],
#         n_iter=20,
#         cv=5,
#         scoring="accuracy",
#         random_state=42,
#         n_jobs=-1,
#         error_score=0,
#         return_train_score=False
#     )


#     random_search.fit(
#         X_train_scaled,
#         y_train
#     )


#     results_random.append({

#         "Model": model_name,

#         "Best Accuracy": random_search.best_score_,

#         "Best Parameters": random_search.best_params_

#     })



# random_results_df = pd.DataFrame(results_random)


# random_results_df = random_results_df.sort_values(
#     by="Best Accuracy",
#     ascending=False
# )


# print(random_results_df)

"""
random_results_df = pd.DataFrame(results_random)
random_results_df.to_csv("random_search_results.csv", index=False)

saved_results = pd.read_csv("random_search_results.csv")

print(saved_results)
"""


grid_save_data = pd.read_csv("grid_search_results.csv")  # this is csv file
# print(f"This is the GRIDSEARCHCV saved data in .csv format : {grid_save_data}")


random_save_data = pd.read_csv("random_search_results.csv")
# print(f"This is RANDOMSEARCHCV saved data in .csv format : {random_save_data}")


# print(f"Final Data completed")




""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
# ================================
# STEP 3: MODEL EVALUATION
# AFTER HYPERPARAMETER TUNING
# ================================
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

# We are using Random Forest is perform well

from sklearn.metrics import accuracy_score, confusion_matrix, classification_report


# Best Random Forest model


# ================================
# LOAD BEST PARAMETERS FROM CSV
# ================================

grid_save_data = pd.read_csv("grid_search_results.csv")

# Get Random Forest row
rf_row = grid_save_data[
    grid_save_data["Model"] == "Random Forest"
]

# print(rf_row)


# Convert string dictionary into python dictionary
import ast

best_params_rf = ast.literal_eval(
    rf_row["Best Parameters"].values[0]
)


# print("Best Parameters:")
# print(best_params_rf)


# Now Recreating Random Forst with best parameter

best_rf = RandomForestClassifier(
    **best_params_rf,
    random_state=42
)


# Train Again
best_rf.fit(
    X_train,
    y_train
)

# Prediction

Y_pred_best_rf  = best_rf.predict(
    X_test
)

"Now Evaluation"

# print(f"\nAccuracy : ""\n", accuracy_score(y_test,Y_pred_best_rf))
# print(f"Confusion Matrix: \n",confusion_matrix(y_test,Y_pred_best_rf))
# print(f"\nClassification Report : \n",classification_report(y_test,Y_pred_best_rf) )



# ================================
# STEP 4: FEATURE IMPORTANCE
# ================================


import pandas as pd
import matplotlib.pyplot as plt


# Get feature importance

importance = best_rf.feature_importances_


# Create dataframe

feature_importance_df = pd.DataFrame({

    "Feature": X.columns,

    "Importance": importance

})


# Sort highest importance

feature_importance_df = feature_importance_df.sort_values(
    by="Importance",
    ascending=False
)


# print(feature_importance_df.head(15))


# Plot top 15 features

plt.figure(figsize=(10,6))


plt.barh(
    feature_importance_df["Feature"].head(15),
    feature_importance_df["Importance"].head(15)
)


# plt.xlabel("Importance")

# plt.ylabel("Features")

# plt.title("Top 15 Features Affecting Churn")


# plt.gca().invert_yaxis()

# plt.show()

# print(df.columns)




""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""'"""
# ================================
# STEP 4: CUSTOMER SEGMENTATION
# K-MEANS CLUSTERING
# ================================
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""



"""
Then why K-Means (Unsupervised)?

Because we are adding a separate analysis part:

Supervised part:

"Which customers will leave?"

Unsupervised part:

"What types/groups of customers exist?"

K-Means does not predict churn. It finds hidden patterns in customers.

Example:

Cluster 1:

High monthly charges
Low tenure
Month-to-month contract
→ High-risk customers

Cluster 2:

Long tenure
Low charges
Yearly contract
→ Loyal customers

So your project becomes:

Prediction + Customer Understanding

Random Forest → Churn prediction ✅
K-Means → Customer segmentation/insights ✅

This makes the project more industry-like because companies want both:

Who may leave?
Which customer group needs attention?
"""

# Importing all required library
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt


"Step 4.1 — Prepare Data for Clustering"

# Copy features
X_cluster = X.copy()


# Scaling is important for K-Means
cluster_scaler = StandardScaler()


X_cluster_scaled = cluster_scaler.fit_transform(
    X_cluster
)

# print(f"X : \n{X}")

# print(f"X_cluster : \n", X_cluster)

# print(f"X_cluster_scaled : \n", X_cluster_scaled)


# print(X_cluster_scaled.shape)



"Step 4.2 — Find Best Number of Clusters (Elbow Method)"


# Finding optimal K value

inertia = []


for k in range(1,11):

    kmeans = KMeans(
        n_clusters=k,
        random_state=42,
        n_init=10
    )


    kmeans.fit(
        X_cluster_scaled
    )


    inertia.append(
        kmeans.inertia_
    )


# Plot

# plt.figure(figsize=(8,5))


# plt.plot(
#     range(1,11),
#     inertia,
#     marker="o"
# )


# plt.xlabel("Number of Clusters")

# plt.ylabel("Inertia")

# plt.title("Elbow Method")


# plt.show()



"Step 4.3 — Apply K-Means and create customer segments"


# ================================
# APPLY FINAL K-MEANS MODEL
# ================================


from sklearn.cluster import KMeans


kmeans = KMeans(
    n_clusters=3,
    random_state=42,
    n_init=10
)


# Train clustering model

clusters = kmeans.fit_predict(
    X_cluster_scaled
)

# Add cluster labels to original data

df["Customer_Segment"] = clusters


# print(df.head())


# print(X)


print(df.columns)
print(df["Customer_Segment"])
# ✔️ 1. Check how many customers in each cluster
print(df["Customer_Segment"].value_counts())



# ✔️ 2. Understand each cluster behavior (MOST IMPORTANT)
print(df.groupby("Customer_Segment").mean(numeric_only=True))

# ✔️ 3. Give meaning to clusters (BUSINESS LABELS)

def label_segment(x):
    if x == 0:
        return "High Value Customers"
    elif x == 1:
        return "Loyal Customers"
    else:
        return "Low Value / Risk Customers"

df["Segment_Label"] = df["Customer_Segment"].apply(label_segment)


# ✔️ 4. Final check
print(df[["Customer_Segment", "Segment_Label"]].value_counts())



# ✅ FINAL MODEL SAVING CODE (Correct for YOUR project)

import joblib

# ================================
# STEP 5: SAVE MODELS + OBJECTS
# ================================

# 1. Save Best Churn Prediction Model (Random Forest)
joblib.dump(best_rf, "churn_model.pkl")

# 2. Save Scaler for prediction models
joblib.dump(scaler, "scaler.pkl")

# 3. Save Cluster Scaler (for K-Means)
joblib.dump(cluster_scaler, "cluster_scaler.pkl")

# 4. Save KMeans model (important for segmentation)
joblib.dump(kmeans, "kmeans_model.pkl")

# 5. Save Label function mapping (optional but useful)
joblib.dump(label_segment, "segment_label_function.pkl")

print("✅ All models saved successfully!")