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

from sklearn.preprocessing import RobustScaler
from sklearn.cluster import AgglomerativeClustering

from sklearn.preprocessing import RobustScaler, StandardScaler
import pickle


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
@st.cache_data   
def load_data(file_name):
    df = pd.read_csv(file_name)
    return df

@st.cache_data  
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
    dims = (len(numbers), 3)
    fig, axes = plt.subplots(dims[0], dims[1], figsize=(12, 10),tight_layout = True)
    axis_i = 0 
    buf = io.BytesIO()
    for c in numbers:
        temp = dataframe[[c]].dropna()
        sns.histplot(data = temp,kde=True,ax=axes[axis_i,0])
        axes[axis_i,0].set_title('Histogram phân phối của '+ str(c))
        sns.boxplot(data = temp, x=c,ax=axes[axis_i,1]) 
        axes[axis_i,1].set_title('Boxplot của '+ str(c))
        sns.boxplot(data = temp, x=c,ax=axes[axis_i,2], showfliers = False) 
        axes[axis_i,2].set_title('Boxplot đã loại bỏ outliers của '+ str(c))
        axis_i +=1
    fig.savefig(buf, format = 'png')
    return buf

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
        hc = AgglomerativeClustering(n_clusters=4, linkage ='ward')
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
st.set_page_config(
    page_title="Customer Segmentation",
    page_icon="🤖",)

st.sidebar.success("Let ML algorithms help you do customer segmentation job")
st.markdown("<h2 style='text-align: center;'>Customer Segmentation with ML algorithm </h2>", unsafe_allow_html=True)
# Load data
# data = load_data_train('train_data/CDNOW_master.csv')
data = load_data_train('train_data/CDNOW_sample.csv')
# Upload file
st.write("""### Load data""")
st.write(""" Tải lên dữ liệu transaction data theo định dạng như hình sau:\n
['customer_id', 'order_date', 'order_quantity', 'order_amounts'] """)
st.image("data_upload_format.png")
uploaded_file_1 = st.file_uploader("Choose a file", type=['csv'])
if uploaded_file_1 is not None:
    data = load_data_train(uploaded_file_1)

st.dataframe(data.head(5))
# st.text(info_dataframe(data))
# Let’s take a closer look at the data we will need to manipulate.
st.code('Transactions timeframe from {} to {}'.format(data['order_date'].min(), data['order_date'].max()))
st.code('{:,} transactions don\'t have a customer id'.format(data[data.customer_id.isnull()].shape[0]))
st.code('{:,} unique customer_id'.format(len(data.customer_id.unique())))
# RFM
data_RFM = calculate_RFM(data)
st.write(""" ### Calculate RFM for each customers """)
st.write('Dữ liệu sau khi tính toán RFM cho',len(data_RFM),'khách hàng'
            ,(data_RFM.head(5)))
st.write('Thông tin của dữ liệu')
st.text(info_dataframe(data_RFM))
fig = visualize_numeric_data(data_RFM, 'customer_id')
st.image(fig)
# Normalization
# st.write("""## Normalization and Scaling""")
# data_RFM = normalize_scaling_dataframe(data_RFM, _scaling_type = RobustScaler()) 
# st.write('Dữ liệu sau khi được chuẩn hoá và scale',data_RFM.head(5))
# st.write('Trực quan hoá dữ liệu sau khi được xử lý')
# fig = visualize_numeric_data(data_RFM, 'customer_id')
# st.image(fig)
st.write("### ML algorithm to do customer segmentation")
st.write("#### Hierarchical")
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
st.write("""### Result Visualization """)

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
    st.markdown("<h3 style='text-align: center;'>Bar chart</h3>", unsafe_allow_html=True)
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
    st.markdown("<h3 style='text-align: center;'>Pie chart</h3>", unsafe_allow_html=True)
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
    st.markdown("<h3 style='text-align: center;'>Tree Map</h3>", unsafe_allow_html=True)
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
    st.markdown("<h3 style='text-align: center;'>Scatter Plot</h3>", unsafe_allow_html=True)
    st.plotly_chart(fig5)
################################################################
elif chart_type == 'Customer Segmentation - 3D Scatter Plot':
    ################################################################
    ## Scatter Plot - 3D
    fig6 = px.scatter_3d(data_RFM, x='Recency', y='Frequency', z='Monetary',
                    color = 'RFM_Cluster', opacity=0.3,color_discrete_map = colors_dict3)
    fig6.update_traces(marker=dict(size=5),
                    selector=dict(mode='markers'))
    st.markdown("<h3 style='text-align: center;'>Scatter Plot 3D </h3>", unsafe_allow_html=True)
    st.plotly_chart(fig6)

#### Export the result
st.write("### Save the result")
st.write("Dữ liệu phân nhóm khách hàng", data_RFM[::150])
# data_save_fn = 'result_data/customer_segment_data.csv'
# data_RFM.to_csv(data_save_fn,index = False).encode('utf-8')
st.download_button(label="Download customer segment data as CSV",
                    file_name='customer_segment_data.csv',
                    mime='text/csv',
                    data=convert_df(data_RFM))
    
