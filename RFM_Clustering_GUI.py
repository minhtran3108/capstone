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


@st.cache_resource
def KNN_best_model(X_train, y_train):
        with st.echo():
            kf = KFold(n_splits=5)
            # Use GridSearchCV to find the best parameters for the models
            from sklearn.model_selection import GridSearchCV
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


#------------------
# Function
@st.cache_data  # 
def load_data(file_name):
    df = pd.read_csv(file_name)
    return df

@st.cache_data  # 
def load_data_train(file_name):
    df = pd.read_csv(file_name)
    # Create transaction index
    df['transaction_index'] = range(1, len(df)+1)
    # Convert order_date to datetime type
    df['order_date'] = df['order_date'].apply(lambda x: pd.to_datetime(x,format='%Y%m%d', errors='coerce'))
    # Remove duplicated rows
    df = df.drop_duplicates().reset_index(drop=True)
    # Remove Null rows
    df  = df.dropna().reset_index(drop=True)
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
# Function to calculate Recency, Frequency, Monetary
@st.cache_data
def calculate_RFM(dataframe):
    # Convert string to date, get max date of dataframe
    max_date = dataframe['order_date'].max().date()
    Recency = lambda x : (max_date - x.max().date()).days
    Frequency  = lambda x: len(x.unique())
    Monetary = lambda x : round(sum(x), 2)
    dataframe_RFM = dataframe.groupby('customer_id').agg({'order_date': Recency,
                                            'transaction_index': Frequency,  
                                            'order_amounts': Monetary }).reset_index()
    # Rename the columns of dataframe
    dataframe_RFM.columns = ['customer_id', 'Recency', 'Frequency', 'Monetary']
    return dataframe_RFM

# Function to normalization and scaling data
@st.cache_data
def normalize_scaling_dataframe(dataframe, _scaling_type = RobustScaler()): # default RobusScaler()
    scaled_features = dataframe.copy()
    scaled_features['R_log'] = np.log1p(scaled_features['Recency'])
    scaled_features['F_log'] = np.log1p(scaled_features['Frequency'])
    scaled_features['M_log'] = np.log1p(scaled_features['Monetary'])
    col_names = ['R_log', 'F_log','M_log']
    ## Scaling
    features = scaled_features[col_names]
    scaler = _scaling_type.fit(features.values)
    features = scaler.transform(features.values)
    new_cols = ['R_sc','F_sc','M_sc']
    dataframe[new_cols] = features
    return dataframe
# Function to apply hierarchical clustering
@st.cache_data
def get_hc_labels(data: pd.DataFrame):
    data = normalize_scaling_dataframe(data, _scaling_type = RobustScaler()) 
    with st.echo():
        hc = AgglomerativeClustering(n_clusters=4, affinity = 'euclidean', linkage ='ward')
    hc.fit(data[['R_sc','F_sc','M_sc']])
    return hc.labels_

# Function get info of dataframe for streamlit
@st.cache_data
def info_dataframe(dataframe):
    buffer = io.StringIO()
    dataframe.info(buf = buffer)
    s = buffer.getvalue()
    return s

# Function to calculate average values and return the size for each segment
@st.cache_data
def calculate_segment(dataframe, col_cluster):
    rfm_agg = dataframe.groupby(col_cluster).agg({
    'Recency': 'mean',
    'Frequency': 'mean',
    'Monetary': ['mean', 'count', 'sum']}).round(0)

    rfm_agg.columns = rfm_agg.columns.droplevel()
    rfm_agg.columns = ['RecencyMean','FrequencyMean','MonetaryMean', 'Quantity','TotalAmount']
    rfm_agg['Percent_Quantity'] = round((rfm_agg['Quantity']/rfm_agg.Quantity.sum())*100, 2)
    rfm_agg['Percent_Amount'] = round((rfm_agg['TotalAmount']/rfm_agg.TotalAmount.sum())*100, 2)
    # Reset the index
    rfm_agg = rfm_agg.reset_index()
    rfm_agg = rfm_agg.sort_values(['MonetaryMean','FrequencyMean', 'RecencyMean'], 
                                    ascending = [False,False,False])
    return rfm_agg
@st.cache_data
def convert_df(df):
    # IMPORTANT: Cache the conversion to prevent computation on every rerun
    return df.to_csv(index = False).encode('utf-8')
#------------------
# GUI
menu = ["Business Objective", "RFM Analysis", "Predict new customer" ]
choice = st.sidebar.selectbox('Menu', menu)
if choice == 'Business Objective':    
    st.markdown("<h1 style='text-align: center; color: black;'>Customer Segmentation</h1>", unsafe_allow_html=True)
    st.subheader("Business Objective")
    st.write("""#### Vấn đề:  
Từ những khách hàng tiêu dùng lớn cho đến những khách hàng rời bỏ doanh nghiệp, 
tất cả những khách hàng đều có nhu cầu và mong muốn đa dạng. Doanh nghiệp muốn khách hàng chi tiêu nhiều hơn
từ những chiến dịch tiếp thị chương trình, sản phẩm mới tới khách hàng theo những cách khác nhau. 
Tuy nhiên, câu hỏi đặt ra là làm thế nào để đưa ra được các chiến dịch tiếp thị phù hợp với những nhóm khách 
hàng đang có nhu cầu để từ đó tăng tỷ lệ phản hồi từ khách hàng và từ đó tăng doanh số bán hàng. 
Bài toán đặt ra là làm thế nào để có thể phân khúc khách hàng một cách tương đối chính xác dựa trên hành vi giao dịch lịch sử
của khách hàng, thuật toán RFM sẽ giúp chúng ta giải quyết vấn đề này một cách nhanh chóng và hiệu quả.""")

    st.write(""" #### - Phân khúc/ nhóm/ cụm khách hàng (market segmentation còn được gọi là phân khúc thị trường) 
    là quá trình nhóm các khách hàng lại với nhau dựa trên các đặc điểm chung. Nó phân chia và nhóm 
    khách hàng thành các nhóm nhỏ theo đặc điểm địa lý, nhân khẩu học, tâm lý học, hành vi (geographic, 
    demographic, psychographic, behavioral) và các đặc điểm khác.""")
    st.image("RFM_Model.png")
    st.write("""  *Read more information about the RFM [here](https://en.wikipedia.org/wiki/RFM_(market_research))*
    """)
    st.write(""" #### - Phân tích RFM (Recency, Frequency, Monetary) 
là một kĩ thuật phân khúc khách hàng dựa trên hành vi giao dịch của khách hàng trong quá khứ 
để nhóm thành các phân khúc.
   
**Dựa trên 3 chỉ số chính:**  
- Recency (R): Thời gian giao dịch cuối cùng.  
- Frequency (F): Tổng số lần giao dịch chi tiêu.
- Monetary value (M): Tổng só tiền giao dịch chi tiêu.  

**Lợi ích của phân tích RFM:**
- Tăng tỷ lệ giữ chân khách hàng.
- Tăng tốc độ phản hồi từ khách hàng.
- Tăng tỷ doanh thu từ khách hàng. """) 
    st.write(""" ##### Mục tiêu: 
    Xây dựng hệ thống phân cụm khách hàng dựa trên các thông tin do công ty cung cấp từ đó có thể giúp công ty xác định
    các nhóm khách hàng khác nhau để có chiến lược kinh doanh, chăm sóc khách hàng phù hợp.
    """)

elif choice == "RFM Analysis":
    st.markdown("<h1 style='text-align: center; color: black;'>Capstone Project</h1>", unsafe_allow_html=True)
    st.markdown("<h2 style='text-align: center; color: black;'>Customer Segmentation</h2>", unsafe_allow_html=True)
    # Read data
    # data = load_data_train('train_data/CDNOW_master.csv')
    data = load_data_train('train_data/CDNOW_sample.csv')
    # Upload file
    st.write("""## Read data""")
    st.write(""" Tải lên dữ liệu transaction data theo định dạng như hình sau:\n
    ['customer_id', 'order_date', 'order_quantity', 'order_amounts'] """)
    st.image("data_upload_format.png")
    uploaded_file = st.file_uploader("Choose a file", type=['csv'])
    if uploaded_file is not None:
        data = load_data_train(uploaded_file)
    
    st.dataframe(data.head(5))
    # st.text(info_dataframe(data))
    # Let’s take a closer look at the data we will need to manipulate.
    st.code('Transactions timeframe from {} to {}'.format(data['order_date'].min(), data['order_date'].max()))
    st.code('{:,} transactions don\'t have a customer id'.format(data[data.customer_id.isnull()].shape[0]))
    st.code('{:,} unique customer_id'.format(len(data.customer_id.unique())))
    # RFM
    data_RFM = calculate_RFM(data)
    st.write(""" ## Calculate RFM for each customers """)
    st.write('Dữ liệu sau khi tính toán RFM cho',len(data_RFM),'khách hàng'
             ,(data_RFM.head(5)))
    st.write('Thông tin của dữ liệu')
    st.text(info_dataframe(data_RFM))
    # fig = visualize_numeric_data(data_RFM, 'customer_id')
    # st.pyplot(fig)
    ## Normalization
    # st.write("""## Normalization and Scaling""")
    # data_RFM = normalize_scaling_dataframe(data_RFM, _scaling_type = RobustScaler()) 
    # st.write('Dữ liệu sau khi được chuẩn hoá và scale',data_RFM.head(5))
    # st.write('Trực quan hoá dữ liệu sau khi được xử lý')
    # fig = visualize_numeric_data(data_RFM, 'customer_id')
    # st.pyplot(fig)
    st.write("## Customer Segmentation")
    st.write("## Hierarchical")
    st.write("Áp dụng thuật toán Hierarchical với số lượng Cluster mong muốn là 4")
    data_RFM["RFM_Cluster"] = get_hc_labels(data_RFM)
    # st.write("Dataframe:",data_RFM)
    rfm_hc_agg = calculate_segment(data_RFM,'RFM_Cluster')
    st.write(rfm_hc_agg,'Kết quả phân cụm theo thuật toán Hierarchical với số lượng nhóm là 4:')
    st.write("""Dựa trên kết quả phân cụm của thuật toán Hierarchical, 
             dữ liệu được phân ra các nhóm (từ trên xuống):  
        - Nhóm 1: Các khách hàng chi tiêu nhiều và thường xuyên, với lượng chi tiêu lớn  
        - Nhóm 2: Các khách hàng chi tiêu và mức độ thường xuyên nằm ở mức khá  
        - Nhóm 3: Các khách hàng chi tiêu ít và không thường xuyên  
        - Nhóm 4: Các khách hàng chi tiếu ít và đã lâu không phát sinh giao dịch.""")
    current_labels = rfm_hc_agg.RFM_Cluster.unique()
    desired_labels = ['STARS','BIG SPENDER','REGULAR','RISK']
    st.write('Đặt tên các nhóm từ trên xuống là:')
    s= ''
    for i in desired_labels:
        s += "- " + i + "\n"
    st.markdown(s)
    # create a dictionary for your corresponding values
    map_dict_hc = dict(zip(current_labels, desired_labels))
    # map the desired values back to the dataframe
    rfm_hc_agg['RFM_Cluster'] = rfm_hc_agg['RFM_Cluster'].map(map_dict_hc)
    # map the desired values back to the data_RFM
    data_RFM['RFM_Cluster'] = data_RFM['RFM_Cluster'].map(map_dict_hc)
    st.dataframe(rfm_hc_agg)
    # Visualization the Result
    colors_dict3 = {'RISK':'yellow','BIG SPENDER':'royalblue',
            'REGULAR':'green', 'STARS':'gold'}
    st.write("""## Visualization the Result""")
    
    ################################################################
    ## Radio box
    chart_type = st.radio(
    "What's kind of visualization graph you want to check?",
    ('Customer Segmentation & Amount Spent',
     'Customer Segmentation - Ration',
     'Customer Segmentation - Tree map', 
     'Customer Segmentation - Scatter Plot',
     'Customer Segmentation - 3D Scatter Plot'))
    ################################################################
    if chart_type == 'Customer Segmentation & Amount Spent':
        ################################################################
        ## BarPlot
        fig1, ax1 = plt.subplots(figsize=(8, 8))
        ax2 = ax1.twinx()
        ax1.bar(rfm_hc_agg['RFM_Cluster'], rfm_hc_agg['Quantity'], color = 'blue')
        ax1.yaxis.set_major_locator(LinearLocator(9))
        # Line Chart
        ax2.plot(rfm_hc_agg['RFM_Cluster'], rfm_hc_agg['TotalAmount'], color = 'red', linestyle ='--')
        # Bar chart setting
        ax1.set_yticklabels(['{:,}'.format(int(x)) for x in ax1.get_yticks().tolist()])
        ax1.set_xlabel("Customer Segment")
        ax1.set_ylabel("Quantity", color='blue', fontsize=14)
        ax1.tick_params(axis="y", labelcolor = 'blue' )
        # Line chart setting
        ax2.yaxis.set_major_locator(LinearLocator(9))
        locator = LinearLocator(numticks=10)
        ax2.set_yticklabels(['{:,}'.format(int(x)) for x in ax2.get_yticks().tolist()])
        ax2.set_ylabel("TotalAmount ($) ", color='red', fontsize=14)
        ax2.tick_params(axis="y", labelcolor='red')
        fig1.suptitle("Customer Segment & Amount Spent - Hierarchical", fontsize=20)
        st.markdown("<h3 style='text-align: center; color: black;'>Bar chart</h3>", unsafe_allow_html=True)
        st.pyplot(fig1)
        fn1 = 'graph/Hierarchical_Customer_SegmentandAmountSpent.png'
        save_graph(fig1, fn1)
    ################################################################
    elif chart_type == 'Customer Segmentation - Ration':
        ################################################################
        ## PiePlot
        fig3, ax3= plt.subplots(figsize=(8, 8))
        ax3 = plt.pie(rfm_hc_agg['Quantity'], autopct='%.2f%%',labels=None)
        plt.title('Tỉ lệ của các cluster',fontsize=13)
        plt.legend(labels=rfm_hc_agg['RFM_Cluster'], loc="upper left")
        plt.axis('off')
        st.markdown("<h3 style='text-align: center; color: black;'>Pie chart</h3>", unsafe_allow_html=True)
        st.pyplot(fig3)
        fn3 = 'graph/Hierarchical_Customer_Segment_Ration_PieChart.png'
        save_graph(fig3, fn3)
    ################################################################
    elif chart_type == 'Customer Segmentation - Tree map':
        ################################################################
        ## Tree Map
        fig4, ax4 = plt.subplots(figsize=(14, 10))
        ax4 = squarify.plot(sizes=rfm_hc_agg['Quantity'],
                    text_kwargs={'fontsize':8,'weight':'bold'},
                    color = colors_dict3.values(),
                    label=['{}\n{:.0f} days \n{:.0f} orders \n{:.0f} $ \nCustomer: {:.0f} ({}%) \nAmount: {:,}({}%)'\
                            .format(rfm_hc_agg.iloc[i][0],rfm_hc_agg.iloc[i][1],rfm_hc_agg.iloc[i][2],rfm_hc_agg.iloc[i][3],
                                    rfm_hc_agg.iloc[i][4],rfm_hc_agg.iloc[i][6],rfm_hc_agg.iloc[i][5],rfm_hc_agg.iloc[i][7])
                            for i in range(0, len(rfm_hc_agg))], alpha=0.5 )

        ax4.set_title("Customers Segments - Hierarchical",fontsize=26,fontweight="bold")
        plt.axis('off')
        st.markdown("<h3 style='text-align: center; color: black;'>Tree Map</h3>", unsafe_allow_html=True)
        st.pyplot(fig4)
        fn4 = 'graph/Hierarchical_Customer_Segment_TreeMap.png'
        save_graph(fig4, fn4)
    ################################################################
    elif chart_type == 'Customer Segmentation - Scatter Plot':
        ################################################################
        ## Scatter Plot
        fig5 = px.scatter(rfm_hc_agg, x="RecencyMean", y="MonetaryMean", size="FrequencyMean",
                        color = "RFM_Cluster", color_discrete_map = colors_dict3,
                        hover_name="RFM_Cluster", size_max=100)
        st.markdown("<h3 style='text-align: center; color: black;'>Scatter Plot</h3>", unsafe_allow_html=True)
        st.plotly_chart(fig5)
    ################################################################
    elif chart_type == 'Customer Segmentation - 3D Scatter Plot':
        ################################################################
        ## Scatter Plot - 3D
        fig6 = px.scatter_3d(data_RFM, x='Recency', y='Frequency', z='Monetary',
                        color = 'RFM_Cluster', opacity=0.3,color_discrete_map = colors_dict3)
        fig6.update_traces(marker=dict(size=5),
                        selector=dict(mode='markers'))
        st.markdown("<h3 style='text-align: center; color: black;'>Scatter Plot 3D </h3>", unsafe_allow_html=True)
        st.plotly_chart(fig6)
    
    ## Export the result
    st.write("# Export the result")
    st.write("Dữ liệu phân nhóm khách hàng", data_RFM[:2357])
    # data_save_fn = 'result_data/customer_segment_data.csv'
    # data_RFM.to_csv(data_save_fn,index = False).encode('utf-8')
    st.download_button(label="Download customer segment data as CSV",
                       file_name='customer_segment_data.csv',
                       mime='text/csv',
                       data=convert_df(data_RFM))
        
elif choice == "Predict new customer":
    st.subheader("Dự đoán khách hàng mới bằng KMeans dựa trên cách phân cụm khách hàng theo thuật toán Hierarchical")
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
    uploaded_file = st.file_uploader("Choose a file", type=['csv'])
    if uploaded_file is not None:
        data = pd.read_csv(uploaded_file)
    st.write('Dữ liệu training cho model KNN:',(data[:50]))
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
    st.write("### Saving Pickle")
    # Save classification model as pickle
    pkl_model_filename = "model/KNN_pickle.pkl"
    with open(pkl_model_filename, 'wb') as file:  
        pickle.dump(model, file)
    st.write("KNN pickle")
    st.download_button("Download KNN Model Pickle", 
                       data=pickle.dumps(model),
                       file_name="KNN_pickle.pkl")
    
    # Save scaling model as  pickle
    pkl_scaling_filename = "model/scaler_pickle.pkl"  
    with open(pkl_scaling_filename, 'wb') as file:  
        pickle.dump(scaler_wrapper, file)
    st.write("Scaling Model pickle")
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
        
    # Inverse map dict
    inverse_map = {map_dict[k] : k for k in map_dict}
    ################################################################
    flag = False
    data_input = None
    type = st.radio("Upload data or Input data?", options=("Upload", "Input"))
    if type=="Upload":
        # Upload file
        st.write(""" Tải lên dữ liệu training phân cụm khách hàng theo định dạng:
        ['customer_id', 'Recency', 'Frequency', 'Monetary']""")
        st.image("data_upload_predict_new_format.png")
        uploaded_file_1 = st.file_uploader("Choose a file", type=['csv'], key = 'Predict')
        if uploaded_file_1 is not None:
            data_input = load_data(uploaded_file_1)
            # st.write(data_input.columns)
            # data_input = data_input[0]     
            flag = True       
    if type=="Input":        
        Recency = st.number_input(label="Input Recency of Customer:")
        Frequency = st.number_input(label="Input Frequency of Customer:")
        Monetary = st.number_input(label="Input Monetary of Customer:")
        if Recency*Frequency*Monetary != 0:
            data_input = pd.DataFrame({'Recency': [Recency],
                                       'Frequency': [Frequency],
                                       'Monetary': [Monetary]})
            flag = True
    
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