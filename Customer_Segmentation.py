# Version 2.0
import warnings
warnings.filterwarnings('ignore')


import streamlit as st

st.set_page_config(
    page_title="Landing Pages",
    page_icon="🤟",)

st.sidebar.success("Select a session you wanna have fun with")

st.markdown("<h1 style='text-align: center;'>Capstone Project</h1>", unsafe_allow_html=True)
st.markdown("<h2 style='text-align: center;'>Customer Segmentation</h2>", unsafe_allow_html=True)
st.subheader("Business Objective")
st.write(""" 
Từ những khách hàng tiêu dùng lớn cho đến những khách hàng rời bỏ doanh nghiệp, 
tất cả những khách hàng đều có nhu cầu và mong muốn đa dạng. Doanh nghiệp muốn khách hàng chi tiêu nhiều hơn
từ những chiến dịch tiếp thị chương trình, sản phẩm mới tới khách hàng theo những cách khác nhau. 
Tuy nhiên, câu hỏi đặt ra là làm thế nào để đưa ra được các chiến dịch tiếp thị phù hợp với những nhóm khách 
hàng đang có nhu cầu để từ đó tăng tỷ lệ phản hồi từ khách hàng và từ đó tăng doanh số bán hàng. 
Bài toán đặt ra là làm thế nào để có thể phân khúc khách hàng một cách tương đối chính xác dựa trên hành vi giao dịch lịch sử
của khách hàng, thuật toán RFM sẽ giúp chúng ta giải quyết vấn đề này một cách nhanh chóng và hiệu quả.""")

st.write(""" #### Phân khúc/ nhóm/ cụm khách hàng (market segmentation còn được gọi là phân khúc thị trường) 
là quá trình nhóm các khách hàng lại với nhau dựa trên các đặc điểm chung. Nó phân chia và nhóm 
khách hàng thành các nhóm nhỏ theo đặc điểm địa lý, nhân khẩu học, tâm lý học, hành vi (geographic, 
demographic, psychographic, behavioral) và các đặc điểm khác.""")
st.image("RFM_Model.png")
st.write("""  *Read more information about the RFM [here](https://en.wikipedia.org/wiki/RFM_(market_research))*
""")
st.write(""" #### Phân tích RFM (Recency, Frequency, Monetary) 
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
st.write(""" #### Mục tiêu: 
Xây dựng hệ thống phân cụm khách hàng dựa trên các thông tin do công ty cung cấp từ đó có thể giúp công ty xác định
các nhóm khách hàng khác nhau để có chiến lược kinh doanh, chăm sóc khách hàng phù hợp.
""")
