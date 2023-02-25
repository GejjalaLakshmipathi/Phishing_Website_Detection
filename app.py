import streamlit as st
import machine_learning as ml
import feature_extraction as fe
from bs4 import BeautifulSoup
import requests as re
import matplotlib.pyplot as plt
from PIL import Image
image = Image.open('images.jpeg')

st.image(image)

# col1, col2 = st.columns([1, 3])
st.title('Final Year Project- 2019-23 Batch, Kalasalingam University')
st.title('Implement a deep learning based system that detects whether a website is fake.')

st.subheader('**Developed By:**')
st.write('GEJJALA LAKSHMIPATHI - 9919005070')
st.write('VARDIREDDY JASWANTH KUMAR REDDY - 9919005224')
st.write('DUGGISETTY BALAMANIKANTA ESWAR - 9919005278')

with st.expander("PROJECT DETAILS"):
    st.subheader('**Approach**')
    st.write('* We have used _supervised learning_ to classify phishing and legitimate websites.')
    st.markdown('* It benefit from content-based approach and we firstly focus on html of the websites.')
    st.markdown('* And also, we used scikit-learn for the ML models.')
    
    st.markdown('* For this project,We created our own data set and defined features, some from the literature and some based on manual analysis.')
    st.markdown('* we used requests library to collect data, BeautifulSoup module to parse and extract features.')
    st.write('The source code and data sets are available in the below Github link:')
    st.write('**_https://github.com/GejjalaLakshmipathi/Phishing_Website_Detection_**')

    st.subheader('Data set')
    st.markdown('* We have used _"phishtank.org"_ & _"tranco-list.eu"_ as data sources.')
    st.write('Totally 26584 websites ==> **_16060_ legitimate** websites | **_10524_ phishing** websites')
    st.write('Data set was created in November 2022.')

    # ----- FOR THE PIE CHART ----- #
    labels = 'phishing', 'legitimate'
    phishing_rate = int(ml.phishing_df.shape[0] / (ml.phishing_df.shape[0] + ml.legitimate_df.shape[0]) * 100)
    legitimate_rate = 100 - phishing_rate
    sizes = [phishing_rate, legitimate_rate]
    explode = (0.1, 0)
    fig, ax = plt.subplots()
    ax.pie(sizes, explode=explode, labels=labels, shadow=True, startangle=90, autopct='%1.1f%%')
    ax.axis('equal')
    st.pyplot(fig)
    # ----- !!!!! ----- #

    st.write('Features + URL + Label ==> Dataframe')
    st.markdown('label is 1 for phishing, 0 for legitimate')
    number = st.slider("Select row number to display", 0, 100)
    st.dataframe(ml.legitimate_df.head(number))


    @st.cache
    def convert_df(df):
        # IMPORTANT: Cache the conversion to prevent computation on every rerun
        return df.to_csv().encode('utf-8')

    csv = convert_df(ml.df)

    st.download_button(
        label="Download data as CSV",
        data=csv,
        file_name='phishing_legitimate_structured_data.csv',
        mime='text/csv',
    )

    st.subheader('Features')
    st.markdown('* Firstly, we have used only content-based features. Still we didn\'t use url-based faetures like length of url etc.')
    st.markdown('* Most of the features extracted using find_all() method of BeautifulSoup module after parsing html.')

    st.subheader('Results')
    st.markdown('* We have used 7 different ML classifiers of **scikit-learn** and tested them implementing **k-fold cross validation**.')
    st.markdown('* Firstly obtained their confusion matrices, then calculated their accuracy, precision and recall scores.')
    st.write('**Comparison table is below:**')
    st.table(ml.df_results)
    st.write('NB --> Gaussian Naive Bayes')
    st.write('SVM --> Support Vector Machine')
    st.write('DT --> Decision Tree')
    st.write('RF --> Random Forest')
    st.write('AB --> AdaBoost')
    st.write('NN --> Neural Network')
    st.write('KN --> K-Neighbours')

with st.expander('LETS CHECK WITH SOME OF THE PHISHING URLs:'):
    st.write('_https://ww3.4movierulz.to/_')
    st.write('_https://form.jotform.com/230093869703865_')
    st.write('_https://www.cryptocurrencystate.net/_')
    st.caption('REMEMBER, PHISHING WEB PAGES HAVE SHORT LIFECYCLE! SO, THE EXAMPLES HERE MENTIONED ARE GIVEN IN DATASET AND EXAMPLES SHOULD BE UPDATED AS WE UPDATE DATASET!')

choice = st.selectbox("Please select your machine learning model",
                 [
                     'Gaussian Naive Bayes', 'Support Vector Machine', 'Decision Tree', 'Random Forest',
                     'AdaBoost', 'Neural Network', 'K-Neighbours'
                 ]
                )

model = ml.nb_model

if choice == 'Gaussian Naive Bayes':
    model = ml.nb_model
    st.write('GNB model is selected!')
elif choice == 'Support Vector Machine':
    model = ml.svm_model
    st.write('SVM model is selected!')
elif choice == 'Decision Tree':
    model = ml.dt_model
    st.write('DT model is selected!')
elif choice == 'Random Forest':
    model = ml.rf_model
    st.write('RF model is selected!')
elif choice == 'AdaBoost':
    model = ml.ab_model
    st.write('AB model is selected!')
elif choice == 'Neural Network':
    model = ml.nn_model
    st.write('NN model is selected!')
else:
    model = ml.kn_model
    st.write('KN model is selected!')


url = st.text_input('Enter the URL')
# check the url is valid or not
if st.button('Check!'):
    try:
        response = re.get(url, verify=False, timeout=4)
        if response.status_code != 200:
            print(". HTTP connection was not successful for the URL: ", url)
        else:
            soup = BeautifulSoup(response.content, "html.parser")
            vector = [fe.create_vector(soup)]  # it should be 2d array, so I added []
            result = model.predict(vector)
            if result[0] == 0:
                st.success("This web page seems a legitimate!")
                st.balloons()
            else:
                st.warning("Attention! This web page is a potential PHISHING!")
                st.snow()

    except re.exceptions.RequestException as e:
        print("--> ", e)





