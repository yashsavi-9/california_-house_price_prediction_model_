import streamlit as st
import pandas as pd
import random
from sklearn.preprocessing import StandardScaler
import pickle
import time

#title
col=['MedInc', 'HouseAge', 'AveRooms', 'AveBedrms', 'Population', 'AveOccup']
st.title('california housing price prediction')
st.image('C:\\Users\\paras\\Downloads\\pexels-binyaminmellish-106399.jpg')
st.header('model of housing prices to predict median house values in california',divider=True)
#st.subheader('''user  must enter given value to predict price:
#['MedInc', 'HouseAge', 'AveRooms', 'AveBedrms', 'Population', 'AveOccup']''')
st.sidebar.title('select house features')
st.sidebar.image('https://media.istockphoto.com/id/1026205392/photo/beautiful-luxury-home-exterior-at-twilight.jpg?s=612x612&w=0&k=20&c=HOCqYY0noIVxnp5uQf1MJJEVpsH_d4WtVQ6-OwVoeDo=')
temp_df=pd.read_csv('california.csv')
random.seed(25)
all_values=[]
for i in temp_df[col]:
    min_value,max_value=temp_df[i].agg(['min','max'])
    var=st.sidebar.slider(f'select {i} range',int(min_value),int(max_value),random.randint(int(min_value),int(max_value)))
    all_values.append(var)
ss=StandardScaler()
ss.fit(temp_df[col])
final_value=ss.transform([all_values])
import pickle
with open('house_price_pred_ridge_model.pkl','rb') as f:
    chatgpt=pickle.load(f)

price=chatgpt.predict(final_value)[0]


import time
st.write(pd.DataFrame(dict(zip(col,all_values)),index=[1]))

progress_bar = st.progress(0)
placeholder = st.empty()
placeholder.subheader('Predicting Price')
place=st.empty()
place.image('https://i.pinimg.com/originals/d7/34/49/d73449313ecedb997822efecd1ee3eac.gif',width=100)

if price>0:

    for i in range(100):
        time.sleep(0.05)
        progress_bar.progress(i + 1)

    body = f'Predicted Median House Price: ${round(price,2)} Thousand Dollars'
    placeholder.empty()
    place.empty()
    # st.subheader(body)

    st.success(body)
else:
    body = 'Invalid House features Values'
    st.warning(body)