import argparse
import os
#from utils.all_utils import read_yaml, create_directory
import shutil
import logging
from flask import request

import pandas as pd

from app.main import db
from app.main.model.prediction import Prediction
from app.main.service.auth_helper import Auth
from app.main.service.utils.all_utils import read_yaml, create_directory
from app.main.util.apiResponse import apiresponse, ApiResponse
from sklearn.model_selection import train_test_split , GridSearchCV
from sklearn.tree import DecisionTreeClassifier
import numpy as np
from sklearn.ensemble import BaggingClassifier
import pickle
import datetime
logging_str = "[%(asctime)s: %(levelname)s: %(module)s]: %(message)s"
log_dir = "logs"
os.makedirs(log_dir, exist_ok=True)
logging.basicConfig(filename=os.path.join(log_dir, 'running_logs.log'), level=logging.INFO, format=logging_str,
                    filemode="a")

config = read_yaml("app/config/config.yaml")
local_data_dirs = config["local_data_dirs"][0]
logging.info(f"reading the data {local_data_dirs}")
file_name = config['file_name'][0]
file_path = os.path.join(local_data_dirs, file_name)
df = pd.read_csv(file_path, sep=',')
x = df.drop(columns='quality')
y = df.quality
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=.25, random_state=30)
def get_data(config_path):
    config = read_yaml(config_path)
    source_download_dirs = config["source_download_dirs"]
    local_data_dirs = config["local_data_dirs"]
    create_directory(local_data_dirs)
    shutil.copy(source_download_dirs[0],local_data_dirs[0])
    logging.info(f"Input data copied successfully from {source_download_dirs} to {local_data_dirs}")


def get_data_new(data):
    try:
        config = read_yaml("app/config/config.yaml")
        input_data_path = data['input_data_path']
        source_download_dirs = None
        if input_data_path:
            source_download_dirs = input_data_path
        else:
            source_download_dirs = config["source_download_dirs"]

        local_data_dirs = config["local_data_dirs"]
        create_directory(local_data_dirs)
        shutil.copy(source_download_dirs, local_data_dirs[0])
        logging.info(f"Input data copied successfully from {source_download_dirs} to {local_data_dirs}")

        apiResponse = ApiResponse(True, "data downloaded successfully", None, None)
        return apiResponse.__dict__, 200

    except Exception as e:
        error = ApiResponse(False, 'something went wrong',
                            None, str(e))
        return (error.__dict__), 500
model_accuracy = {}
def train_decission_tree():
    # config = read_yaml("app/config/config.yaml")
    # local_data_dirs = config["local_data_dirs"][0]
    # logging.info(f"reading the data {local_data_dirs}")
    # file_name = config['file_name'][0]
    # file_path = os.path.join(local_data_dirs,file_name)
    # df = pd.read_csv(file_path,sep=',')
    # x = df.drop(columns='quality')
    # y = df.quality
    #x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=.25, random_state=30)
    dt_model = DecisionTreeClassifier()
    dt_model.fit(x_train, y_train)
    df1 = df.head(500)
    x1 = df1.drop(columns='quality')
    y1 = df1.quality
    dt_model1 = DecisionTreeClassifier()
    dt_model1.fit(x1, y1)
    path = dt_model1.cost_complexity_pruning_path(x1, y1)
    ccp_alpha = path.ccp_alphas
    dt_modle2 = []
    for ccp in ccp_alpha:
        dt_m = DecisionTreeClassifier(ccp_alpha=ccp)
        dt_m.fit(x1, y1)
        dt_modle2.append(dt_m)

    dt_model2 = []
    score = []
    best_score = 0
    best_ccp_alpha = 0
    for i in ccp_alpha:
        dt_m = DecisionTreeClassifier(ccp_alpha=i)
        dt_m.fit(x1, y1)
        dt_model2.append(dt_m)
        #score.append(dt_m.score(x_test, y_test))
        if best_score < dt_m.score(x_test, y_test):
            best_score = dt_m.score(x_test, y_test)
            best_ccp_alpha = i
            score.append(dt_m.score(x_test, y_test))
    dt_model_ccp = DecisionTreeClassifier(random_state=0, ccp_alpha=best_ccp_alpha)
    dt_model_ccp.fit(x1, y1)
    grid_pram = {"criterion": ['gini', 'entropy'],
                 "splitter": ['best', 'random'],
                 "max_depth": range(2, 40, 1),
                 "min_samples_split": range(2, 10, 1),
                 "min_samples_leaf": range(1, 10, 1),
                 'ccp_alpha': np.random.rand(20)
                 }
    #grid_ccp = GridSearchCV(estimator=dt_model_ccp, param_grid=grid_pram, verbose=1,cv = 10, n_jobs = -1)
    #grid_ccp.fit(x1, y1)
    #logging.info(grid_ccp.best_params_)
    # get the best params
    max_depth = config['estimators']['DecisionTreeClassifier']['params']['max_depth']
    criterion = config['estimators']['DecisionTreeClassifier']['params']['criterion']
    ccp_alpha = config['estimators']['DecisionTreeClassifier']['params']['ccp_alpha']
    min_samples_leaf = config['estimators']['DecisionTreeClassifier']['params']['min_samples_leaf']
    min_samples_split = config['estimators']['DecisionTreeClassifier']['params']['min_samples_split']
    splitter = config['estimators']['DecisionTreeClassifier']['params']['splitter']
    dt_cpp_new = DecisionTreeClassifier(criterion=criterion, max_depth=max_depth, min_samples_leaf=min_samples_leaf, min_samples_split=min_samples_split,splitter=splitter, ccp_alpha=ccp_alpha)
    dt_cpp_new.fit(x1, y1)
    logging.info("Decission Tree Scores:- ",dt_cpp_new.score(x_test, y_test))
    print("training done")
    model_accuracy['decission_tree']=dt_cpp_new.score(x_test, y_test)
    saved_model_dir = config['saved_model_dir']
    create_directory(saved_model_dir)
    with open(saved_model_dir[0] + '/decission_tree.sav', 'wb') as f:
        pickle.dump(dt_cpp_new, f)



from sklearn.ensemble import RandomForestClassifier
def train_byRandomForest():
    rf = RandomForestClassifier(n_estimators=10)
    rf.fit(x_train, y_train)
    rf.score(x_test, y_test)
    model_accuracy['rf']=rf.score(x_test, y_test)
    saved_model_dir = config['saved_model_dir']
    create_directory(saved_model_dir)
    with open(saved_model_dir[0] + '/rf.sav', 'wb') as f:
        pickle.dump(rf, f)

def train_byBagging():
    bag_dt = BaggingClassifier(DecisionTreeClassifier(),n_estimators=10)
    bag_dt.fit(x_train,y_train)
    logging.info("score achive by Bagging:- ",bag_dt.score(x_test, y_test))
    model_accuracy['bag_dt']=bag_dt.score(x_test, y_test)
    saved_model_dir = config['saved_model_dir']
    create_directory(saved_model_dir)
    with open(saved_model_dir[0] + '/bag_dt.sav', 'wb') as f:
        pickle.dump(bag_dt, f)


from sklearn.neighbors import KNeighborsClassifier
def train_byKNN():
    bag_knn = BaggingClassifier(KNeighborsClassifier(6),n_estimators=10)
    bag_knn.fit(x_train,y_train)
    logging.info("score achive by Bagging of KNN:- ",bag_knn.score(x_test, y_test))
    model_accuracy['bag_knn']=bag_knn.score(x_test, y_test)
    saved_model_dir = config['saved_model_dir']
    create_directory(saved_model_dir)
    with open(saved_model_dir[0] + '/bag_knn.sav', 'wb') as f:
        pickle.dump(bag_knn, f)
def train():
    try:
        resp, status = Auth.get_logged_in_user(request)
        user = resp['data']
        if user['role'] == 'admin':
            train_decission_tree()
            train_byRandomForest()
            train_byBagging()
            train_byKNN()
            Keymax = max(zip(model_accuracy.values(), model_accuracy.keys()))[1]
            print(Keymax)
            del model_accuracy[Keymax]
            saved_model = config['saved_model_dir'][0]
            for key in model_accuracy.keys():
                os.remove(saved_model + "/" + key + ".sav")

            print(model_accuracy)

            apiResponse = ApiResponse(True, "model trained successfully", None, None)
            return apiResponse.__dict__, 200
        else:
            apiResponse = ApiResponse(True, "you don't have the admin privilage to train the model", None, None)
            return apiResponse.__dict__, 400

    except Exception as e:
        error = ApiResponse(False, 'there is some problem while training', None,
                            str(e))
        return (error.__dict__), 500



def predict(data):
    try:
        resp, status = Auth.get_logged_in_user(request)
        user = resp['data']
        if user['role'] == 'admin':
            current_time = datetime.datetime.utcnow()
            fixed_acidity = data['fixed_acidity']
            volatile_acidity = data['volatile_acidity']
            citric_acid = data['citric_acid']
            residual_sugar = data['residual_sugar']
            chlorides = data['chlorides']
            free_sulfur_dioxide = data['free_sulfur_dioxide']
            total_sulfur_dioxide = data['total_sulfur_dioxide']
            density = data['density']
            pH = data['pH']
            sulphates = data['sulphates']
            alcohol = data['alcohol']
            prediction = Prediction(
                user_id = user['id'],
                fixed_acidity = fixed_acidity,
                volatile_acidity = volatile_acidity,
                citric_acid = citric_acid,
                residual_sugar = residual_sugar,
                chlorides = chlorides,
                free_sulfur_dioxide = free_sulfur_dioxide,
                total_sulfur_dioxide = total_sulfur_dioxide,
                density = density,
                pH = pH,
                sulphates = sulphates,
                alcohol = alcohol,
                created_at = current_time
            )
            try:
                db.session.add(prediction)
                db.session.commit()
            except Exception as e:
                db.session.rollback()
                apiResponce = ApiResponse(
                    False, 'Error Occurd',
                    'null', f'Database {str(prediction)} : {str(e)}')
                return (apiResponce.__dict__), 500

            # Load the Model back from file
            with open('app/model/bag_dt.sav', 'rb') as file:
                model_file = pickle.load(file)

            prediction_value = model_file.predict([[fixed_acidity,volatile_acidity,citric_acid,residual_sugar,chlorides,free_sulfur_dioxide,total_sulfur_dioxide,density,pH,sulphates,alcohol]])
            print(prediction_value)
            print(prediction_value[0])
            prediction = Prediction.query.filter_by(id=prediction.id).first()
            #id = prediction.id,
            prediction.quality = int(prediction_value[0])

            try:
                db.session.commit()
            except Exception as e:
                db.session.rollback()
                apiResponce = ApiResponse(
                    False, 'Error Occurd',
                    'null', f'Database {str(prediction)} : {str(e)}')
                return (apiResponce.__dict__), 500
        apiResponse = ApiResponse(True, f"prediction successfull, prediction:- {prediction_value} ", None, None)
        return apiResponse.__dict__, 200


    except Exception as e:
        error = ApiResponse(False, 'there is some problem while training', None,
                            str(e))
        return (error.__dict__), 500

if __name__ == '__main__':
    args = argparse.ArgumentParser()

    args.add_argument("--config", "-c", default="app/config/config.yaml")

    parsed_args = args.parse_args()

    try:
        logging.info("\n\n\n>>>>> stage one started")
        get_data(config_path=parsed_args.config)
        logging.info("stage one completed! all the data are saved in local >>>>>")
    except Exception as e:
        #logging.exception(e)
        raise e


