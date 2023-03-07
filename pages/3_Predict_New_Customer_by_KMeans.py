import warnings
warnings.filterwarnings('ignore')

import streamlit as st
import io
import pandas as pd
import numpy as np
import seaborn as sns
import plotly.express as px
import squarify
import matplotlib.pyplot as plt
from matplotlib.ticker import LinearLocator
from matplotlib.figure import Figure

from feature_engine.wrappers import SklearnTransformerWrapper
from sklearn.preprocessing import RobustScaler
from sklearn.cluster import AgglomerativeClustering

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import RobustScaler, StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import KFold
from sklearn.model_selection import GridSearchCV

from streamlit_yellowbrick import st_yellowbrick
from yellowbrick.classifier import ConfusionMatrix, ClassificationReport, ROCAUC, roc_auc

import pickle

#------------------
# Function

@st.cache_resource
def KNN_best_model(X_train, y_train):
        with st.echo():
            kf = KFold(n_splits=5)
            # Use GridSearchCV to find the best parameters for the models
            # Create a list of parameters of Logistic Regression for the GridSearchCV
            k_range = [6, 10 ,15, 20, 25]
            param_grid = dict(n_neighbors=k_range)
            # Create a list of models to test
            clf_grid = GridSearchCV(KNeighborsClassifier(), param_grid, cv=kf, n_jobs=-1)
            search_clf = clf_grid.fit(X_train, y_train)
            best_clf = search_clf.best_estimator_
            # Build model with best Parameter
            best_model = KNeighborsClassifier(n_neighbors=clf_grid.best_params_['n_neighbors'])
            model = best_model.fit(X_train, y_train)
        return model
    
@st.cache_resource
def load_pickle(pkl_filename):
    with open(pkl_filename, 'rb') as file:  
        model = pickle.load(file)
    return model


def save_graph(plot: Figure, file_name):
    plot.savefig(file_name)
    with open(file_name, "rb") as img:
        st.download_button(
            label="Download Graph",
            data=img,
            file_name=file_name,
            mime="image/png")



@st.cache_data  
def load_data(file_name):
    df = pd.read_csv(file_name)
    return df

@st.cache_data
def visualize_numeric_data(dataframe,drop_columns):
    numbers = dataframe.drop([drop_columns], axis =1).select_dtypes(['int16', 'int32', 'int64', 'float16', 'float32', 'float64']).columns
    # Visualize continuous vars of the data
    dims = (len(numbers), 2)
    fig, axes = plt.subplots(dims[0], dims[1], figsize=(12, 10),tight_layout = True)
    axis_i = 0 
    for c in numbers:
        temp = dataframe[[c]].dropna()
        sns.histplot(data = temp,kde=True,ax=axes[axis_i,0])
        axes[axis_i,0].set_title('Biểu đồ Histogram phân phối của '+ str(c))
        sns.boxplot(data = temp, x=c,ax=axes[axis_i,1]) 
        axes[axis_i,1].set_title('Biểu đồ boxplot của '+ str(c))
        axis_i +=1
    return fig

# Function get info of dataframe for streamlit
@st.cache_data
def info_dataframe(dataframe):
    buffer = io.StringIO()
    dataframe.info(buf = buffer)
    s = buffer.getvalue()
    return s


@st.cache_data
def convert_df(df):
    # IMPORTANT: Cache the conversion to prevent computation on every rerun
    return df.to_csv(index = False).encode('utf-8')
#------------------
# GUI
st.set_page_config(
    page_title="Predict New Customer",
    page_icon="🌟",)

st.sidebar.success("Now, you can predict your new customer")

st.markdown("<h1 style='text-align: center;'>Capstone Project</h1>", unsafe_allow_html=True)
st.markdown("<h2 style='text-align: center;'>Customer Segmentation</h2>", unsafe_allow_html=True)
st.subheader("Predict new customer by KMeans based on Hierarchical Clustering")
current_labels = ['STARS','BIG SPENDER','REGULAR','RISK']
# Upload file
st.write("""## Read data""")
data = load_data('train_data/customer_segment_data_master.csv')
st.write("### Training data")
st.write(""" Tải lên dữ liệu training phân cụm khách hàng theo định dạng:\n
['customer_id', 'Recency', 'Frequency', 'Monetary', 'RFM_Cluster']""")
st.write('Với các nhóm khách hàng như sau:')
s= ''
for i in current_labels:
    s += "- " + i + "\n"
st.markdown(s)
st.write("Dataframe theo format sau:")
st.image("data_upload_training_predict_new_format.png")
uploaded_file_2 = st.file_uploader("Choose a file", type=['csv'])
if uploaded_file_2 is not None:
    data = load_data(uploaded_file_2)
st.write('Dữ liệu training cho model KNN:',(data[:50]))
st.write('Thông tin của dữ liệu')
st.text(info_dataframe(data))
## Convert category into numeric for target column
desired_labels = [0,1,2,3]
map_dict = dict(zip(current_labels, desired_labels))
st.write('Chuyển đổi cột dữ liệu phân loại sang kiểu số', map_dict)
data['target'] = data['RFM_Cluster'].map(map_dict)
# Code
# Build Model with KNN
st.write("## Build Model with KNN and find the best ParaMeter with GRIDSEARCHCV")
X = data.drop(['customer_id', 'RFM_Cluster', 'target'], axis = 1)
y = data.target
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.3, random_state=42)
## Scaling data
st.write("### Chuẩn hoá dữ liệu bằng Standard Scaler")
with st.echo():
    scaler_wrapper = SklearnTransformerWrapper(StandardScaler()) 
    X_train = scaler_wrapper.fit_transform(X_train, ['Recency', 'Frequency', 'Monetary'])
    X_test = scaler_wrapper.transform(X_test)

## Build Model
### GridSearch to find best Parameter
st.write("### Find the best ParaMeter with GridSearchCV")
# Fit the best model to the training data, fit to train data
model = KNN_best_model(X_train, y_train)    
### Accuracy
st.write("### Accuracy")
train_accuracy = accuracy_score(y_train,model.predict(X_train))*100
test_accuracy = accuracy_score(y_test,model.predict(X_test))*100
st.code(f'Train accuracy: {round(train_accuracy,2)}% \nTest accuracy: {round(test_accuracy,2)}%')
st.markdown("**Model KNN hoạt động phù hợp trên tập dữ liệu**")

st.write("## Result Report")    
chart_type = st.radio(
"Result Report",
('Classification Report',
'Confusion Matrix',
'ROC CURVE',))
if chart_type == 'Confusion Matrix':        
    st.write("### Confusion Matrix")
    ### Confusion Matrix
    cm = ConfusionMatrix(model, classes=y.unique())
    # Fit fits the passed model. This is unnecessary if you pass the visualizer a pre-fitted model
    cm.fit(X_train, y_train)
    cm.score(X_test, y_test)
    st_yellowbrick(cm)
### Classification Report
if chart_type == 'Classification Report':        
    st.write("### Classification Report")
    clr = ClassificationReport(model, classes=y.unique(), support=True)
    clr.fit(X_train, y_train)        
    clr.score(X_test, y_test)        
    st_yellowbrick(clr)   
### ROC AUC 
if chart_type == 'ROC CURVE':
    st.write('### ROC CURVE')
    rocauc = ROCAUC(model, classes=y.unique()) 
    rocauc.fit(X_train, y_train)        
    rocauc.score(X_test, y_test,)        
    st_yellowbrick(rocauc)  
st.markdown("**Thuật toán nhận diện tốt nhóm khách hàng tiềm năng là STARS và BIGSPENDER**")
################################
# Save classification model as pickle
pkl_model_filename = "model/KNN_pickle.pkl"
with open(pkl_model_filename, 'wb') as file:  
    pickle.dump(model, file)
    
# Save scaling model as  pickle
pkl_scaling_filename = "model/scaler_pickle.pkl"  
with open(pkl_scaling_filename, 'wb') as file:  
    pickle.dump(scaler_wrapper, file)   
st.write("### Saving Pickle")
col1, col2, = st.columns(2)
with col1:
    st.markdown("##### KNN Model Pickle")
    st.download_button("Download KNN Model Pickle", 
                    data=pickle.dumps(model),
                    file_name="KNN_pickle.pkl")

with col2:
    st.markdown("##### Scaling Model pickle")
    st.download_button("Download Scaling Model Pickle", 
                    data=pickle.dumps(scaler_wrapper),
                    file_name="scaler_pickle.pkl")
    
################################################################
### Predict New 
st.write("## Predict New Customer")
# Load scaling model pickle
scaling_model = load_pickle("model/scaler_pickle.pkl")
# Load model pickle
classification_model = load_pickle("model/KNN_pickle.pkl")
# st.write("### Loading Model File")    
# col1, col2, = st.columns(2)
# with col1:
#     st.markdown("##### KNN Model Pickle")
#     knn_file = st.file_uploader("Upload Model Predict", type=['pkl'], key = 'Model file')
#     if knn_file is not None:
#         classification_model = load_pickle(knn_file)
# with col2:
#     st.markdown("##### Scaling Model pickle")
#     scalinng_file = st.file_uploader("Upload Scaling Pickle", type=['pkl'], key = 'Scaling file')
#     if scalinng_file is not None:
#         classification_model = load_pickle(scalinng_file)

    
# Inverse map dict
inverse_map = {map_dict[k] : k for k in map_dict}
################################################################
flag = False
data_input = None
type = st.radio("Upload data or Input data?", options=( "Input","Upload data"))
if type=="Upload data":
    # Upload file
    st.write(""" Tải lên dữ liệu training phân cụm khách hàng theo định dạng:
    ['customer_id', 'Recency', 'Frequency', 'Monetary']""")
    st.image("data_upload_predict_new_format.png")
    uploaded_file_3 = st.file_uploader("Choose a file", type=['csv'], key = 'predict')
    if uploaded_file_3 is not None:
        data_input = load_data(uploaded_file_3)
        # st.write(data_input.columns)
        # data_input = data_input[0]     
        flag = True       
if type=="Input":
    col1, col2, col3 = st.columns(3)
    with col1:
        Recency = st.number_input(label="Input Recency of Customer:", value = 100)
    with col2:
        Frequency = st.number_input(label="Input Frequency of Customer:", value = 2)
    with col3:
        Monetary = st.number_input(label="Input Monetary of Customer:", value = 200)
    if Recency*Frequency*Monetary != 0:
        data_input = pd.DataFrame({'Recency': [Recency],
                                    'Frequency': [Frequency],
                                    'Monetary': [Monetary]})
        flag = True

if st.button('Run Predict New'):
    if flag:
        st.markdown("**Input:**")
        st.dataframe(data_input)
        x_new = scaling_model.transform(data_input[['Recency','Frequency','Monetary']])
        data_input['predict'] = classification_model.predict(x_new)
        data_input['predict'] = data_input['predict'].map(inverse_map)
        st.markdown("**Result:**")
        st.dataframe(data_input)
        st.download_button(label="Download predicted data as CSV",
                    file_name='predicted_data.csv',
                    mime='text/csv',
                    data=convert_df(data_input))
    else:
        st.write("Please insert the data input to predict")