# Databricks notebook source
# MAGIC %md
# MAGIC ##Overview
# MAGIC This notebook consists simple demonstartion of mlflow integration in machine learning model. We have implemented 4 different classifier model
# MAGIC i.e Random Forest, Logistic Regression, KNN, Decision Tree and we are tracking it using Mlflow. <br/><br/>
# MAGIC Mlflow tracking helps in: 
# MAGIC * model logging
# MAGIC * log parameters
# MAGIC * log metrics
# MAGIC * log artifacts

# COMMAND ----------

pip install databricks-cli

# COMMAND ----------

# MAGIC %md
# MAGIC ####Importing the libraries

# COMMAND ----------

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# COMMAND ----------

# MAGIC %md
# MAGIC ####Data Preprocessing steps

# COMMAND ----------

class data_analysis:
    
    @staticmethod
    def load_data(path):
        '''
        this function is to read the csv file
        Parameters:
        path(string): input the dataset location
        Returns: returns a dataframe 
        '''
        data = pd.read_csv(path)
        return data
    
    @staticmethod
    def data_cleaning(data):
        '''
        this function is to check if any null values are there and to drop them if it's present.
        Parameters:
        data: input the dataframe
        Returns: returns the dataframe having no null values
        '''
        print("na values available in data \n")
        print(data.isna().sum())
        data = data.dropna()
        print("after droping na values \n")
        print(data.isna().sum())
        return data
    
    @staticmethod
    def preprocessing1(data):
        from sklearn.preprocessing import StandardScaler,OneHotEncoder
        from sklearn.impute import SimpleImputer
        from sklearn.compose import ColumnTransformer
        from sklearn.pipeline import Pipeline
        numeric_features = data.select_dtypes('number').columns
        categorical_features = data.select_dtypes('object').columns
        numeric_transformer = Pipeline(steps=[("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())])
        categorical_transformer = Pipeline(
        steps=[("imputer", SimpleImputer(strategy="constant", fill_value="missing")),
               ("onehot", OneHotEncoder(handle_unknown="ignore"))])
       
        if len(categorical_features)==0:
            print("has no categorical columns")
            col_transformer = ColumnTransformer(transformers=[("numeric", numeric_transformer,numeric_features)],remainder='passthrough')
            transformed_data = col_transformer.fit_transform(data)
            columns = numeric_features.tolist()
            print(columns)
        else:
            col_transformer = ColumnTransformer(transformers=[("numeric", numeric_transformer, numeric_features),                                                    ("categorical",categorical_transformer,categorical_features)],remainder='passthrough')
            transformed_data = col_transformer.fit_transform(data)
            onehot_cols = (col_transformer.named_transformers_["categorical"].named_steps["onehot"].get_feature_names(categorical_features))
            #print(onehot_cols)
            columns = numeric_features.tolist() + onehot_cols.tolist()
            print(columns)

        #transformed = col_transformer.transform(transformed_data)
        final_data=pd.DataFrame(transformed_data, columns=columns)
        return final_data
    
     
    def train_test_split(final_data):
        '''
        this function is to split the dataset intro training and testing
        Parameters:
        final_data: the dataframe
        Returns: returns a testing data and training data
        '''
        from sklearn.model_selection import train_test_split
        X= final_df.iloc[:,:-2].values

        y = final_df.iloc[:, -1:].values
       
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.7,stratify = y, random_state=47)

        return X_train, X_test, y_train, y_test
    
    @staticmethod
    def predict_on_test_data(model,X_test):
        '''
        this function is used to predict the values of the testing dataset (X) on the model interpreted before
        Parameters:
        model: the classifier model 
        X_test: the remaining data(testing set) comprising of all the columns other than target column
        Returns: returns the predicted values on the testing dataset
        '''
        y_pred = model.predict(X_test)
        return y_pred
    
    @staticmethod
    def predict_prob_on_test_data(model,X_test):
        '''
        this function is used to predict the probabilites of the testing dataset (X) on the model interpreted before
        Parameters:
        model: the classifier model 
        X_test: the remaining testing set comprising of all the columns other than target column
        Returns: returns the probability on the predicted values of the testing dataset
        '''
        y_prob = model.predict_proba(X_test)
        return y_prob
    
    @staticmethod
    def get_metrics(y_test, y_pred,y_prob):
        '''
        this function is used to evaluate certain metrics on the model 
        Parameters:
        y_test: the remaining testing set comprising of only the target column 
        y_pred: the predicted values on the testing dataset
        y_prob: the probability of the predicted values of the testing dataset
        Returns: returns the metrics such as Accuracy, precision, recall and entropy
        '''
        from sklearn.metrics import accuracy_score,precision_score,recall_score,log_loss,f1_score,classification_report
        class_report=classification_report(y_test, y_pred,output_dict=True)
        print(class_report)
#         recall_0 = class_report['0.0']['recall']
#         f1_score_0 = class_report['0.0']['f1-score']
#         precision_0=class_report['0.0']['precision']
#         recall_1 = class_report['1.0']['recall']
#         f1_score_1 = class_report['1.0']['f1-score']
#         precision_1=class_report['1.0']['precision']
        acc = accuracy_score(y_test, y_pred)
        entropy = log_loss(y_test, y_prob)
        return {'accuracy': round(acc, 2), 'entropy': round(entropy, 2)}
#         return {'accuracy': round(acc, 2), 'entropy': round(entropy, 2),'f1_score_0':
#         round(f1_score_0,2),'f1_score_1':
#         round(f1_score_1,2),'recall_0':round(recall_0,2),'recall_1':round(recall_1,2),'precision_0':
#         round(precision_0,2),'precision_1':
#         round(precision_1,2)}
        
    @staticmethod
    def create_roc_auc_plot(y_test,y_prob):
        '''
        this function is used to create the roc-auc plot 
        Parameters:
        clf: the classifier model 
        X_test: the remaining testing set comprising of all the columns other than target column
        y_test: the remaining testing set comprising of only the target column 
        Returns: returns the roc-auc plot
        '''
        
        from sklearn.metrics import roc_curve
        from sklearn.metrics import auc
        

        # Compute fpr, tpr, thresholds and roc auc
        fpr, tpr, thresholds = roc_curve(y_test,y_prob[::,1])
        roc_auc = auc(fpr, tpr)

        # Plot ROC curve
        plt.plot(fpr, tpr, label='ROC curve (area = %0.3f)' % roc_auc)
        plt.plot([0, 1], [0, 1], 'k--')  # random predictions curve
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.0])
        plt.xlabel('False Positive Rate or (1 - Specifity)')
        plt.ylabel('True Positive Rate or (Sensitivity)')
        plt.title('Receiver Operating Characteristic')
        plt.legend(loc="lower right")
        plt.savefig("roc_auc_curve.png")

    @staticmethod
    def create_confusion_matrix_plot(y_test,y_pred):
        '''
        this function is used to create the confusion matrics 
        Parameters:
        y_test: the remaining testing set comprising of only the target column 
        y_pred: the predicted values on the testing dataset
        Returns: returns the confusion matrics and printing the classification report
        '''
        from sklearn.metrics import confusion_matrix
        LABELS = ["0", "1"]
        matrix = confusion_matrix(y_true=y_test, y_pred=y_pred)
        print(matrix)
        plt.figure(figsize=(8, 8))
        confusion_plot=sns.heatmap(matrix, xticklabels=LABELS, yticklabels=LABELS, annot=True, fmt="d");
        #confusion_plot=sns.heatmap(matrix, annot=True, fmt="d");
        plt.title("Confusion matrix")
        plt.ylabel('True class')
        plt.xlabel('Predicted class')
        plt.show()
        fig = confusion_plot.get_figure()
        fig.savefig('confusion_matrix.png')
        

# COMMAND ----------

databricks configure --token

# COMMAND ----------

# MAGIC %md
# MAGIC ####MLFlow Integration

# COMMAND ----------

import mlflow.sklearn
import mlflow
class classification():
        
    def __init__(self, model, params={}):
        """
        Constructor for RandamForestClassifier
        :param params: parameters for the constructor such as no of estimators, depth of the tree, random_state etc
        """
        self.model = model
        self._params = params
             
    @classmethod
    def new_instance(cls, model, params={}):
        return cls(model,params)
            
    def create_experiment(self,runname,confusion_matrix_path = None, 
                      roc_auc_plot_path = None):
        with mlflow.start_run(run_name=runname):
            run = mlflow.active_run()
            #ID = run.info.run_uuid
            run_id = run.info.run_id
            #print(run_id)
            experimentID = run.info.experiment_id
            X_train, X_test, y_train, y_test = data_analysis.train_test_split(final_df)
            # train and predict
            self.model.fit(X_train, y_train)
            y_pred = data_analysis.predict_on_test_data(self.model,X_test)
            #print(y_pred)
            y_prob = data_analysis.predict_prob_on_test_data(self.model,X_test)
           # print(y_prob)
            run_metrics = data_analysis.get_metrics(y_test, y_pred,y_prob)
            print(run_metrics)
            data_analysis.create_confusion_matrix_plot(y_test,y_pred)
            data_analysis.create_roc_auc_plot(y_test,y_prob) 
            mlflow.log_params(self._params)
            for metric in run_metrics:
                mlflow.log_metric(metric, run_metrics[metric])
        
            mlflow.sklearn.log_model(self.model, "model")
        
            if not confusion_matrix_path == None:
                mlflow.log_artifact(confusion_matrix_path, 'confusion_matrix')

            if not roc_auc_plot_path == None:
                mlflow.log_artifact(roc_auc_plot_path, "roc_auc_plot")
            return(experimentID,run_id),runname
            

# COMMAND ----------

# MAGIC %md
# MAGIC ####Loading dataset

# COMMAND ----------


data=data_analysis.load_data("/dbfs/FileStore/tables/datset_lg/banking.csv")
data

# COMMAND ----------

# MAGIC %md
# MAGIC ####Performing EDA

# COMMAND ----------

final_df = data_analysis.preprocessing1(data)
final_df

# COMMAND ----------

# MAGIC %md
# MAGIC ##Random Forest classifier

# COMMAND ----------

from sklearn.ensemble import RandomForestClassifier
params = {"n_estimators": 100, "random_state":1}
model = RandomForestClassifier(**params)
rfr = classification.new_instance(model,params)
(experimentID, run_id),runname=rfr.create_experiment('Random forest','confusion_matrix.png', 'roc_auc_curve.png')
print("MLflow Run completed with run_id {} and experiment_id {}".format(run_id, experimentID))

# COMMAND ----------

# MAGIC %md
# MAGIC ##Logistic Regression

# COMMAND ----------

from sklearn.linear_model import LogisticRegression
params = {"max_iter" : 10000, "random_state":1}
model = LogisticRegression(**params)
lgr = classification.new_instance(model,params)
(experimentID, run_id),runname=lgr.create_experiment('logistic','confusion_matrix.png', 'roc_auc_curve.png')
print("MLflow Run completed with run_id {} and experiment_id {}".format(run_id, experimentID))

# COMMAND ----------

# MAGIC %md
# MAGIC ##K Nearest Neighbors classifier

# COMMAND ----------

from sklearn.neighbors import KNeighborsClassifier
params = {"n_neighbors":50}
model = KNeighborsClassifier(**params)
lgr = classification.new_instance(model,params)
(experimentID, run_id),runname=lgr.create_experiment('KNN','confusion_matrix.png', 'roc_auc_curve.png')
print("MLflow Run completed with run_id {} and experiment_id {}".format(run_id, experimentID))

# COMMAND ----------

# MAGIC %md
# MAGIC ##Decision Tree classifier

# COMMAND ----------

from sklearn.tree import DecisionTreeClassifier
params = {"criterion":'entropy'}
model = DecisionTreeClassifier(**params)
dc = classification.new_instance(model,params)
(experimentID, run_id),runname=dc.create_experiment('Decision tree','confusion_matrix.png', 'roc_auc_curve.png')
print("MLflow Run completed with run_id {} and experiment_id {}".format(run_id, experimentID))

# COMMAND ----------

# MAGIC %md
# MAGIC ##Registering the model
# MAGIC The MLflow Model Registry component is a centralized model store, set of APIs, and UI, to collaboratively manage the full lifecycle of MLflow Models. It provides model lineage (which MLflow Experiment and Run produced the model), model versioning, stage transitions, annotations, and deployment management.

# COMMAND ----------

model_name=runname
model_name

# COMMAND ----------

artifact_path = "model"
model_uri = f'runs:/{run_id}/{artifact_path}.format(run_id=run_id, artifact_path=artifact_path)'
import mlflow
registry_uri = f'databricks://modelregistery:modelregistery'
mlflow.set_registry_uri(registry_uri)
registry_uri

# COMMAND ----------

model_uri

# COMMAND ----------

model_name

# COMMAND ----------

model_details = mlflow.register_model(model_uri=model_uri, name=model_name)

# COMMAND ----------

artifact_path = "model"
model_uri = "runs:/{run_id}/{artifact_path}".format(run_id=run_id, artifact_path=artifact_path)
import mlflow
registry_uri = f'databricks://my_prod_scope:central_model_registry'
mlflow.set_registry_uri(registry_uri)


# COMMAND ----------

model_details = mlflow.register_model(model_uri=model_uri, name=model_name)

# COMMAND ----------

mr_uri = mlflow.get_registry_uri()
print(mr_uri)

# COMMAND ----------

model_uri

# COMMAND ----------

import time
from mlflow.tracking.client import MlflowClient
from mlflow.entities.model_registry.model_version_status import ModelVersionStatus

def wait_until_ready(model_name, model_version):
    client = MlflowClient()
    for _ in range(10):
        model_version_details = client.get_model_version(name=model_name,
        version=model_version,)
        status = ModelVersionStatus.from_string(model_version_details.status)
        print("Model status: %s" % ModelVersionStatus.to_string(status))
        if status == ModelVersionStatus.READY:
            break
        time.sleep(1)
wait_until_ready(model_details.name, model_details.version)

# COMMAND ----------

from mlflow.tracking.client import MlflowClient
client = MlflowClient()

model_version_infos = client.search_model_versions("name = '%s'" % model_name)
new_model_version = max([model_version_info.version for model_version_info in model_version_infos])

wait_until_ready(model_name, new_model_version)

# COMMAND ----------

client.update_model_version(
  name=model_name,
  version=new_model_version,
  description="This model version is a random forest containing 100 decision trees that was trained in scikit-learn."
)

# COMMAND ----------


