import warnings
warnings.filterwarnings('ignore')

import streamlit as st
import io
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.ticker import LinearLocator
from matplotlib.figure import Figure

if 'run' not in st.session_state:
    st.session_state['run'] = False


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
def data_segment(data, R_low, R_high ,\
    F_low, F_high , M_low, M_high ):
    selected_data = \
        data.query('{}<=Recency<={} & {}<=Frequency<={} & {}<=Monetary<={}'.\
        format(R_low, R_high, F_low, F_high, M_low, M_high))
    selected_data['RFM_Cluster'] = 'Selected'
    df_out = pd.merge(data, selected_data, how='left', left_on= data.columns.tolist()
                      , right_on=data.columns.tolist())
    df_out.RFM_Cluster.fillna("Other", inplace = True)
    return df_out

@st.cache_data
def convert_df(df):
    # IMPORTANT: Cache the conversion to prevent computation on every rerun
    return df.to_csv(index = False).encode('utf-8')
#------------------
# GUI
st.set_page_config(
    page_title="Customer Segmentation",
    page_icon="👋",)

st.sidebar.success("This session is where you can looking for valued customers by RFM Values")

st.markdown("<h1 style='text-align: center;'>Capstone Project</h1>", unsafe_allow_html=True)
st.markdown("<h2 style='text-align: center;'>Customer Segmentation</h2>", unsafe_allow_html=True)
# Read data
# data = load_data_train('train_data/CDNOW_master.csv')
data = load_data_train('train_data/CDNOW_sample.csv')
# Upload file
st.write("""## Read data""")
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
st.markdown("## Calculate RFM Values")
data_RFM = calculate_RFM(data)
st.write(""" ## Calculate RFM for each customers """)
st.write('Dữ liệu sau khi tính toán RFM cho',len(data_RFM),'khách hàng'
            ,(data_RFM.head(5)))
st.write('Thông tin của dữ liệu')
st.text(info_dataframe(data_RFM))
st.write('Trực quan hoá các thông số RFM')
fig = visualize_numeric_data(data_RFM, 'customer_id')
st.image(fig)
st.markdown("## Choose customer by RFM Values")
with st.container(): 
    min = int(data_RFM.Recency.min())
    max = int(data_RFM.Recency.max())
    options = list(range(min, max, 1))
    R_low, R_high = st.select_slider(
        'Select a range value for Recency',
        options = options,
        # value=(min, options[-1]))
        value=(min, 170))
    st.write('Recency range values:', R_low,'to', R_high)
    min = int(data_RFM.Frequency.min())
    max = int(data_RFM.Frequency.max())
    options = list(range(min, max, 2))
    F_low, F_high = st.select_slider(
        'Select a range value for Frequency',
        options = options,
        value=(min, options[-1]))
    st.write('Frequency range values:', F_low,'to', F_high)
    min = int(data_RFM.Monetary.min())
    max = int(data_RFM.Monetary.max())
    options = list(range(min, max, 10))
    M_low, M_high = st.select_slider(
        'Select a range value for Monetary',
        options=options,
        value=(min, options[-1]))
    st.write('Monetary range values:', M_low,'to', M_high)
    
if st.button('Run') and st.session_state['run'] == False:
    st.session_state['run'] = True

if st.session_state['run'] == True:
    data_RFM = data_segment(data_RFM, R_low, R_high, F_low, F_high,
        M_low, M_high)
    rfm_agg = calculate_segment(data_RFM, 'RFM_Cluster')
    st.markdown("**RFM Mean Values of selected data:**")
    st.dataframe(rfm_agg)
    ################################################################
    st.write('## Visualization')
    ################################################################
    ## BarPlot
    fig1, ax1 = plt.subplots(figsize=(8, 8))
    ax2 = ax1.twinx()
    ax1.bar(rfm_agg['RFM_Cluster'], rfm_agg['Quantity'], color = 'blue')
    ax1.yaxis.set_major_locator(LinearLocator(9))
    # Line Chart
    ax2.plot(rfm_agg['RFM_Cluster'], rfm_agg['TotalAmount'], color = 'red', linestyle ='--')
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

    ################################################################
    ## PiePlot
    fig3, ax3= plt.subplots(figsize=(8, 8))
    ax3 = plt.pie(rfm_agg['Quantity'], autopct='%.2f%%',labels=None)
    plt.title('Tỉ lệ của các cluster',fontsize=13)
    plt.legend(labels=rfm_agg['RFM_Cluster'], loc="upper left")
    plt.axis('off')
    st.markdown("<h3 style='text-align: center;'>Pie chart</h3>", unsafe_allow_html=True)
    st.pyplot(fig3)

    ## Export the result
    st.write("## Save the selected data")
    data_save = data_RFM[data_RFM['RFM_Cluster']=='Selected']
    data_save = data_save.drop('RFM_Cluster', axis = 1).reset_index(drop = True)
    st.write("Dữ liệu khách hàng được lựa chọn", data_save[::10])
    st.write("Số lượng khách hàng được chọn là:", data_save.shape[0])
    # data_save_fn = 'result_data/valued_customers.csv'
    # data_RFM.to_csv(data_save_fn,index = False).encode('utf-8')
    st.download_button(label="Download customer segment data as CSV",
                        file_name='valued_customers.csv',
                        mime='text/csv',
                        data=convert_df(data_save))
