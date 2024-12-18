import os
import pickle
import pandas as pd
import streamlit as st

from sklearn import metrics
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.ensemble import VotingClassifier

from xgboost import XGBClassifier
from lightgbm.sklearn import LGBMClassifier
# 将页面放大至适应web宽度
st.set_page_config(layout="wide")


# 定义检查Excel文件的函数以便于后续调用
def check_sheet_exists(file_path, sheet_name):
    try:
        xls = pd.ExcelFile(file_path)
        sheet_names = xls.sheet_names
        if sheet_name in sheet_names:
            return True
        else:
            return False
    except:
        return False


# 第一页：项目说明
def page_1():
    st.header('111')
    with st.container(border=True):
        st.subheader('Background')
        st.write('111')
    with st.container(border=True):
        st.subheader('2. NOTES')
        st.write('111')


# 第二页：上传训练数据
def page_2():
    st.header('Data upload')
    # 上传训练数据
    with st.container(border=True):
        st.subheader('Upload training data')
        uploaded_file = st.file_uploader("Please upload Excel data as requested in **NOTES**.")
        with st.expander('**NOTES**'):
            st.write('1. Please specify the outcome variable and name it as **outcome**.')
            st.write('2. Please do not upload data that contains missing values or outliers.')
            st.write('3. You can upload any input variables that you prefer without being limited to the 7 clinical features mentioned in our paper, **but you will not be able to use the *3. Model Prediction* function if you do so**.')
            st.write('4. We sincerely hope that you will try the *3. Model Prediction* function. Therefore, please name the input variables in your Excel as follows: **HCO3, GCS, WBC, INR, HCT, Temperature, BUN**.')
    # 显示已经上传的数据
    with st.container(border=True):
        st.subheader('Loading data')
        if uploaded_file is not None:
            st.success('Uploaded successfully!', icon="✅")
            # 转换为dataframe格式准备储存为excel以供后续调用
            with st.spinner('Wait a moment...'):
                dataframe = pd.read_excel(uploaded_file)
                # 创建缓存文件夹
                if not os.path.exists('Cache'):
                    os.mkdir('Cache')
                # 将上传的数据保存于缓存目录
                dataframe.to_excel('Cache/raw data.xlsx', sheet_name='raw')
                st.write(dataframe)
        else:
            st.info('Please upload training data!', icon="ℹ️")


# 第三页：模型训练
def page_3():
    st.header('Model training')
    with st.container(border=True):
        st.subheader('Training data')
        with st.spinner('Training data has been detected! Load...'):
            if check_sheet_exists(file_path='Cache/raw data.xlsx',
                                  sheet_name='raw'):
                # 从缓存目录中读取数据
                data = pd.read_excel('Cache/raw data.xlsx', sheet_name='raw')
                data = data.drop(data.columns[0], axis=1)
                # st.write('**The raw data values are as follows:**')
                # st.write(data)
                # 划分输入和结局数据
                data = data.dropna()
                y = data['outcome']
                X = data.drop(columns=['outcome'], axis=1)
                X_names = X.columns
                # 输入数据归一化
                scaler = preprocessing.MinMaxScaler()
                scaled_X = scaler.fit_transform(X)
                scaled_X = pd.DataFrame(scaled_X)
                # 合并输入和结局数据
                scaled_data = pd.concat([scaled_X, y], axis=1)
                
                y_name = ['outcome']
                data_names = pd.concat([pd.Series(X_names), pd.Series(y_name)],
                                       axis=0)
                scaled_data.columns = data_names
                # 导出归一化数据
                scaled_data.to_excel('Cache/normalized data.xlsx',
                                     sheet_name='scaled')
                # st.write('**The normalized data values are as follows:**')
                # st.write(scaled_data)
                with st.spinner('Wait a moment......'):
                    st.success("Data has been uploaded. Let's train the model!", icon="✅")
            else:
                st.error("No training data! Please upload the training data on the **1. Data Upload** page!", icon="🚨")    
    with st.container(border=True):
        st.subheader('Ensembel model')
        st.caption("**NOTE:** In order to simplify the training process so that more users can use it, this web application does not support the hyperparameter tuning function at present. However, you can edit the source code to implement it. Maybe you would like to get some guidelines: https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html#sklearn.model_selection.GridSearchCV")
        if st.button("Click to start training! If you are not satisfied, you can click again to train again!", use_container_width=True):
            if check_sheet_exists(file_path='Cache/normalized data.xlsx',
                                  sheet_name='scaled'):
                with st.spinner('Training...'):
                    # 从缓存目录中读取数据
                    scaled_data = pd.read_excel('Cache/normalized data.xlsx',
                                                sheet_name='scaled')
                    scaled_data = scaled_data.drop(scaled_data.columns[0],
                                                   axis=1)
                    # 划分数据
                    data_train, data_test = train_test_split(scaled_data,
                                                             test_size=0.2)
                    y_train = data_train['outcome']
                    X_train = data_train.drop(['outcome'], axis=1)
                    y_test = data_test['outcome']
                    X_test = data_test.drop(['outcome'], axis=1)
                    # 训练模型
                    estimators = [('XGB', XGBClassifier()),
                                  ('LGBM', LGBMClassifier())]
                    model = VotingClassifier(estimators=estimators,
                                             voting='soft')
                    model = model.fit(X_train, y_train)
                    # 测试模型
                    y_pred = model.predict(X_test)
                    y_pred_prob = model.predict_proba(X_test)[:, 1]
                    # 计算性能指标
                    accuracy = metrics.accuracy_score(y_test, y_pred)
                    precision = metrics.precision_score(y_test, y_pred)
                    recall = metrics.recall_score(y_test, y_pred)
                    f1 = metrics.f1_score(y_test, y_pred)
                    AUC = metrics.roc_auc_score(y_test, y_pred_prob)
                    # 保留三位小数
                    accuracy = "%.3f" % accuracy
                    precision = "%.3f" % precision
                    recall = "%.3f" % recall
                    f1 = "%.3f" % f1
                    AUC = "%.3f" % AUC
                    # 显示测试性能
                    st.success('Model training succeeded!', icon="✅")
                    st.write(f'The accuracy in the test cohort is {accuracy}.')
                    st.write(f'The precision in the test cohort is {precision}.')
                    st.write(f'The recall in the test cohort is {recall}.')
                    st.write(f'The F1 score in the test cohort is {f1}.')
                    st.write(f'The AUC in the test cohort is {AUC}.')
                    # 保存训练好的模型
                    with open('Cache/model.pkl', 'wb') as f:
                        pickle.dump(model, f)
            else:
                st.error("No training data! Please upload the training data on the **1. Data Upload** page!", icon="🚨")


# 第四页：模型预测
def page_4():
    st.header('Model prediction')
    # 加载训练好的模型
    if os.path.exists('Cache/model.pkl'):
        st.success("Model has been trained. Let's use the model to make predictions!", icon="✅")
        # 准备提交表单
        with st.form('my_form'):
            st.subheader('Enter data')
            st.caption('Please enter the relevant information and then press **Predict**.')
            # 7个变量
            if check_sheet_exists(file_path='Cache/raw data.xlsx',
                                  sheet_name='raw'):
                # 从缓存目录中读取数据
                raw_data = pd.read_excel('Cache/raw data.xlsx',
                                         sheet_name='raw')
                raw_data = raw_data.drop(raw_data.columns[0], axis=1)
            else:
                st.error("No training data! Please upload the training data on the **1. Data Upload** page!", icon="🚨")    
            # 判断列名是否符合要求
            expected_columns = ['HCO3', 'GCS', 'WBC', 'INR', 'HCT', 'Temperature', 'BUN']
            if set(raw_data.columns)!= set(expected_columns):
                st.error("The data does not meet the requirements for use, please upload according to **NOTES**!", icon="🚨")
                submitted = st.form_submit_button('Unusable')
                if submitted:
                    st.toast("**Don't click!**", icon='🎉')
                    st.stop()
            else:
                HCO3 = st.number_input('HCO3',
                                       min_value=raw_data['HCO3'].min(),
                                       max_value=raw_data['HCO3'].max(),
                                       step=0.1)
                GCS = st.number_input('GCS',
                                      min_value=raw_data['GCS'].min(),
                                      max_value=raw_data['GCS'].max(),
                                      step=0.1)
                WBC = st.number_input('WBC',
                                      min_value=raw_data['WBC'].min(),
                                      max_value=raw_data['WBC'].max(),
                                      step=0.1)
                INR = st.number_input('INR',
                                      min_value=raw_data['INR'].min(),
                                      max_value=raw_data['INR'].max(),
                                      step=0.1)
                HCT = st.number_input('HCT',
                                      min_value=raw_data['HCT'].min(),
                                      max_value=raw_data['HCT'].max(),
                                      step=0.1)
                TEM = st.number_input('Temperature',
                                      min_value=raw_data['Temperature'].min(),
                                      max_value=raw_data['Temperature'].max(),
                                      step=0.1)
                BUN = st.number_input('BUN',
                                      min_value=raw_data['BUN'].min(),
                                      max_value=raw_data['BUN'].max(),
                                      step=0.1)
                # 设置提交按钮
                submitted = st.form_submit_button('Predict')
    else:
        
        st.error("No trained model! Please train the model on the **2. Model Training** page!", icon="🚨")
        
        
    with st.container(border=True):
        st.subheader('Prediction')
        if submitted:
            input_data = pd.DataFrame({'HCO3': [HCO3], 'GCS': [GCS],
                                       'WBC': [WBC], 'INR': [INR],
                                       'HCT': [HCT], 'Temperature': [TEM],
                                       'BUN': [BUN]})

            def norm(x, xmin, xmax):
                x = (x - xmin)/(xmax-xmin)
                return x

            with st.spinner('Wait a moment......'):
                if check_sheet_exists(file_path='Cache/raw data.xlsx',
                                      sheet_name='raw'):
                    # 从缓存目录中读取数据
                    raw_data = pd.read_excel('Cache/raw data.xlsx',
                                             sheet_name='raw')
                    raw_data = raw_data.drop(raw_data.columns[0], axis=1)
                else:
                    st.error("No training data! Please upload the training data on the **1. Data Upload** page!", icon="🚨")

                input_data['HCO3'] = norm(input_data['HCO3'],
                                          raw_data['HCO3'].min(),
                                          raw_data['HCO3'].max())
                input_data['GCS'] = norm(input_data['GCS'],
                                         raw_data['GCS'].min(),
                                         raw_data['GCS'].max())
                input_data['WBC'] = norm(input_data['WBC'],
                                         raw_data['WBC'].min(),
                                         raw_data['WBC'].max())
                input_data['INR'] = norm(input_data['INR'],
                                         raw_data['INR'].min(),
                                         raw_data['INR'].max())
                input_data['HCT'] = norm(input_data['HCT'],
                                         raw_data['HCT'].min(),
                                         raw_data['HCT'].max())
                input_data['Temperature'] = norm(input_data['Temperature'],
                                                 raw_data['Temperature'].min(),
                                                 raw_data['Temperature'].max())
                input_data['BUN'] = norm(input_data['BUN'],
                                         raw_data['BUN'].min(),
                                         raw_data['BUN'].max())
                with open('Cache/model.pkl', 'rb') as f:
                    model = pickle.load(f)
                prediction = model.predict(input_data)
                probability = model.predict_proba(input_data)[:, 1]
                probability = f"{probability[0] * 100:.2f} %"
                if prediction[0] == 0:
                    st.subheader(f'This patient has a low death risk with probability of {probability}.')
                else:
                    st.warning('**Warning! Caution!**', icon="⚠️")
                    st.subheader(f'This patient is at high death risk with probability of {probability}!')
        else:
            st.write('Please complete all the steps above.')
# 整合所用页
pg = st.navigation([st.Page(page_1, title="0. Introduction"),
                    st.Page(page_2, title="1. Data Upload"),
                    st.Page(page_3, title="2. Model Training"),
                    st.Page(page_4, title="3. Make Prediction")])
pg.run()
