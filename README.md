# Bank Marketing Campaign Prediction



This project aims to predict whether a customer will respond positively to a bank's marketing campaign. The dataset used for this project is the "Bank Marketing" dataset, which contains various features related to customer demographics, contact details, and previous campaign outcomes.



## Steps Involved



1. **Data Loading and Preprocessing**:

    - The dataset is loaded using `pandas`.

    - Missing values in categorical columns are filled with the mode of the respective columns.

    - Outliers are detected and removed using the Interquartile Range (IQR) method for columns like 'age' and 'duration'.

    - Categorical columns are encoded using mappings and `LabelEncoder`.



2. **Exploratory Data Analysis (EDA)**:

    - Various plots are created using `matplotlib`, `seaborn`, and `plotly` to visualize the distribution of data and detect outliers.

    - Pie charts, box plots, scatter plots, and count plots are used to understand the data better.



3. **Feature Engineering**:

    - New features are created using transformations like reciprocal, square root, and logarithmic transformations.

    - The dataset is balanced using upsampling techniques to handle class imbalance.



4. **Model Training and Evaluation**:

    - The dataset is split into training and testing sets using `train_test_split`.

    - Models like `RandomForestClassifier`, `DecisionTreeClassifier`, and `XGBClassifier` are trained and evaluated.

    - Metrics like accuracy, F1 score, and precision are calculated to evaluate the models.

    - Confusion matrix, ROC curve, and Precision-Recall curve are plotted for further evaluation.



5. **Model Saving and Deployment**:

    - The trained model is saved using `joblib` for future use.

    - A chatbot interface is created to interact with users and predict their response to the campaign based on their input.



## Tools and Libraries Used



- **Pandas**: For data manipulation and analysis.

- **NumPy**: For numerical operations.

- **Matplotlib and Seaborn**: For data visualization.

- **Plotly**: For interactive plots.

- **Scikit-learn**: For machine learning algorithms and evaluation metrics.

- **XGBoost**: For gradient boosting classifier.

- **Joblib**: For saving and loading the trained model.

- **Pickle**: For serializing and deserializing the model.

- **Time**: For creating a delay in the chatbot responses.



This project demonstrates the complete workflow of a machine learning project, from data preprocessing to model deployment, providing valuable insights and predictions for the bank's marketing campaign.
