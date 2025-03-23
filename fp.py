### IMPORTING LIBRARIES
import warnings     #warnings library to avoid potentisl warnings
warnings.filterwarnings('ignore')
import streamlit as st      #framework libraries
import numpy as np      #librariy to work with arrays
import pandas as pd     #pandas for using DATAFRAME
import seaborn as sns       #genrate plot for analysis
import matplotlib.pyplot as plt     #To print plot in frontend using st
from imblearn.over_sampling import SMOTE        #used in case of inbalance in dataset
from deep_translator import GoogleTranslator        #primary translater used convert multiple languages
from sklearn.model_selection import train_test_split        # to split x and y from data for model
from sklearn.preprocessing import StandardScaler        #scaling values to reduce domination of datas
from sklearn.linear_model import LinearRegression, LogisticRegression  #importing regression and classification model
from sklearn.naive_bayes import GaussianNB      #one of the classification model
from sklearn.svm import SVC     #one of the classification model
from sklearn.neighbors import KNeighborsClassifier    #one of the classification model  
from sklearn.cluster import KMeans      #one of the clustering model  
from sklearn.metrics import confusion_matrix, mean_absolute_error, r2_score, classification_report,accuracy_score,f1_score,mean_squared_error       #mathematical calculation for analysis
from transliterate import translit      #translater for romanization of text
import base64       #to wrok with image in streamlit
import speech_recognition as sr     # voice to text analyser
from gtts import gTTS       #text to voice converter
import plotly.express as px     #create intractive plot
### HOME PAGE SETUP
st.set_page_config(     # initial page configuration
    page_title="Multilingual Translator",
    page_icon="🔠",
    layout="wide",
    initial_sidebar_state="expanded")
### DEFINING NESSESSARY FUNCTION
@st.cache_resource      #using cache
### BACKGROUND IMAGE
def set_background(image_file):
    with open(image_file, "rb") as image:
        encoded_string = base64.b64encode(image.read()).decode()        #decoge image
    st.markdown(        #apply css to set image as background
        f"""
        <style>
        .stApp {{
            background-image: url(data:image/jpg;base64,{encoded_string});
            background-size: cover;
            background-position: center;
            background-repeat: no-repeat;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )
### SCALING DATA FOR PERFECT FIT
def scale_features(X_train, X_test):
    scaler = StandardScaler()         # using standerd scallar
    X_train_scaled = scaler.fit_transform(X_train)  # scaing x train
    X_test_scaled = scaler.transform(X_test)    # scaing x test
    return X_train_scaled, X_test_scaled
### GENERATE DATA FOR ACCURACY CHECKING
def generate_dummy(num_samples=100):
    embeddings = np.random.rand(num_samples, 768)  # Dummy sentence embeddings (768 features)
    labels = np.random.choice([0, 1, 2], size=num_samples)      # giving choice to select
    return embeddings, labels
### ROMANIZING OUTPUT TEXT
def transliterate_text(text, lang_code):
    try:
        return translit(text, lang_code, reversed=True)  # Reverse to get Romanization
    except Exception as e:
        return f"Error in transliteration: {str(e)}"        # print error
### CONVERTING TRANSLATED TEXT TO AUDIO 
def text_to_audio(text, lang):
    try:
        tts = gTTS(text=text, lang=lang, slow=False)  # slow=False ensures faster speech
        tts.save("translated_audio.mp3")        # SAVE AUDIO AS FILE
        return "translated_audio.mp3"
    except Exception as e:
        st.error(f"Error generating audio: {str(e)}")   #PRINT ERROR
        return None
### AVILABLE LANGUAGES
language_map = {        #   languages with language code 
    'Afrikaans': 'af', 'Albanian': 'sq', 'Arabic': 'ar', 'Armenian': 'hy',
    'Azerbaijani': 'az', 'Basque': 'eu', 'Belarusian': 'be', 'Bengali': 'bn',
    'Bosnian': 'bs', 'Bulgarian': 'bg', 'Catalan': 'ca', 'Chinese (Simplified)': 'zh-CN',
    'Chinese (Traditional)': 'zh-TW', 'Croatian': 'hr', 'Czech': 'cs', 'Danish': 'da',
    'Dutch': 'nl', 'English': 'en', 'Esperanto': 'eo', 'Estonian': 'et', 'Filipino': 'tl',
    'Finnish': 'fi', 'French': 'fr', 'Galician': 'gl', 'Georgian': 'ka', 'German': 'de',
    'Greek': 'el', 'Gujarati': 'gu', 'Haitian Creole': 'ht', 'Hebrew': 'iw', 'Hindi': 'hi',
    'Hungarian': 'hu', 'Icelandic': 'is', 'Indonesian': 'id', 'Irish': 'ga', 'Italian': 'it',
    'Japanese': 'ja', 'Javanese': 'jw', 'Kannada': 'kn', 'Kazakh': 'kk', 'Khmer': 'km',
    'Korean': 'ko', 'Kurdish (Kurmanji)': 'ku', 'Lao': 'lo', 'Latin': 'la', 'Latvian': 'lv',
    'Lithuanian': 'lt', 'Macedonian': 'mk', 'Malay': 'ms', 'Malayalam': 'ml', 'Maltese': 'mt',
    'Marathi': 'mr', 'Mongolian': 'mn', 'Nepali': 'ne', 'Norwegian': 'no', 'Persian': 'fa',
    'Polish': 'pl', 'Portuguese': 'pt', 'Punjabi': 'pa', 'Romanian': 'ro', 'Russian': 'ru',
    'Serbian': 'sr', 'Sinhala': 'si', 'Slovak': 'sk', 'Slovenian': 'sl', 'Spanish': 'es',
    'Sundanese': 'su', 'Swahili': 'sw', 'Swedish': 'sv', 'Tamil': 'ta', 'Telugu': 'te',
    'Thai': 'th', 'Turkish': 'tr', 'Ukrainian': 'uk', 'Urdu': 'ur', 'Uzbek': 'uz',
    'Vietnamese': 'vi', 'Welsh': 'cy', 'Xhosa': 'xh', 'Yiddish': 'yi', 'Yoruba': 'yo',
    'Zulu': 'zu'
}
### CHECKING BALANCE IN DATA
def is_imbalanced(y_train, threshold=0.2):
    class_counts = np.bincount(y_train)     # getting count
    total = len(y_train)
    class_ratios = class_counts / total     #finding ratio
    min_class_ratio = np.min(class_ratios)
    return min_class_ratio < threshold
### CREATING SESSIONS
if "is_recording" not in st.session_state:
    st.session_state["is_recording"] = False
if "input_text" not in st.session_state:
    st.session_state["input_text"] = ""
if "final_sentence" not in st.session_state:
    st.session_state["final_sentence"] = ""
### FUNCTION TO RECORD AUDIO
def record_audio():
    r = sr.Recognizer()
    st.write("Recording... Speak now!")
    with sr.Microphone() as source:     #GETTING SORCE FROM MICROPHONE
        audio = r.listen(source)        #listern text
    try:
        text = r.recognize_google(audio)    # recoganize audio
        st.success(f"Recognized Text: {text}")      #print text
        return text
    except sr.UnknownValueError:
        st.error("Sorry, I could not understand the audio.")    #print error
        return ""
    except sr.RequestError as e:
        st.error(f"Could not request results; {e}")     #print error
        return ""
### PRINTING DATA
def generate_dummy_data(num_samples=100):
    np.random.seed(42)
    data = pd.DataFrame({  #createing features
        'Feature_1': np.random.rand(num_samples),
        'Feature_2': np.random.rand(num_samples) * 2,
        'Feature_3': np.random.rand(num_samples) * 3,
        'Target': np.random.choice([0, 1, 2], size=num_samples)
    })
    return data  #getting data as matrix
set_background("C:/Users/diva1/OneDrive/Pictures/finalpp.jpg")      #paste image path 
### transator
translator = GoogleTranslator()     #initiaize translater function
### PAGE FUNCTION
st.title("🔠 Translation Evaluation with Multiple ML Models")  #show title
selected_model = st.sidebar.radio("EXPLORE 🔎", ['🎖️ Clustering', '🎖️ Regression', '🎖️ Classification'])        #type of model to exicute
selected_language = st.sidebar.selectbox("Select Target Language: 🚀", list(language_map.keys())) # language selection
selected_language_code = language_map[selected_language]        #getting language code for selected language
st.subheader("🎙️ Simple Voice-to-Text Feature",divider='red')       #show subheader
### input function
if st.button("Record"):     #record audio
    if st.session_state["is_recording"]:
        # Stop recording
        st.session_state["is_recording"] = False
        st.session_state["input_text"] = record_audio()  # Store voice input
    else:
        # Start recording
        st.session_state["is_recording"] = True
st.write('---')
type_text= st.text_input("Enter a sentence: 👇", "")        # get text fom user
if type_text:       # select audio orr typed text
    st.session_state["final_sentence"] = type_text
elif st.session_state["input_text"]:
    st.session_state["final_sentence"] = st.session_state["input_text"]
st.write("Final Input Sentence: ", st.session_state["final_sentence"]) # final text 
input_sentence=st.session_state["final_sentence"] # stoe text for translation
### CORE FUNCTION
if input_sentence:
    translated_sentence = GoogleTranslator(source='auto', target=selected_language_code).translate(input_sentence)  #TRANSLATING SENTENCE
    st.write('---')
    st.subheader(f"**🎰 Translated Sentence ({selected_language}):** {translated_sentence}")  # SHOW TRANSLATED SENTENCE 
    st.write('---')
    romanized_sentence = transliterate_text(translated_sentence, selected_language_code)        #ROMANIZE TRANSLATED SENTENCE
    st.subheader(f"**📈 Romanized Sentence:** {romanized_sentence}",divider='red')  # SHOW SENTENCE
    st.write("Hear Treanslated Audio: 👇", "")
    audio_file = text_to_audio(translated_sentence, selected_language_code)     # STORE AUDIO FILE
    if audio_file:
        audio_data = open(audio_file, 'rb').read()      #read audio file
        st.audio(audio_data, format='audio/mp3')        #play audio
    else:
        st.error("Could not generate audio for the translated text.")
### MODEL SELECTION AND FITTING
X, y = generate_dummy()     # get datas 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)       #split x and y for modal  
if is_imbalanced(y_train):  #CHECK BALENCE 
    smote = SMOTE()     #APPLAY SMOTE AND FIT DATA
    X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)
else:
    st.write("Dataset is balanced, no need for SMOTE.")
    X_train_smote, y_train_smote = X_train, y_train
X_train_scaled, X_test_scaled = scale_features(X_train_smote, X_test) # SCALE DATA FOR BETTER OUTPUT
### MODEL SELECTION
if selected_model == '🎖️ Clustering':
    st.header("Clustering Task with K-Means",divider='red')
    model = KMeans(n_clusters=3) # use kmeans as model in clustering
    model.fit(X_train_scaled)
    predictions = model.predict(X_test_scaled) # fit and predict using model
elif selected_model == '🎖️ Regression':
    st.header("Regression Task with Linear Regression",divider='red')
    model = LinearRegression()      # use LR for regression 
    model.fit(X_train_scaled, y_train)
    predictions = model.predict(X_test_scaled).round()      # fit and predict using model
elif selected_model == '🎖️ Classification':
    st.header("Classification Task",divider='red')
    models = {      #declere all classification model
        'Naive Bayes': GaussianNB(),
        'SVM': SVC(),
        'KNN': KNeighborsClassifier(n_neighbors=5),
        'LR':LogisticRegression(max_iter=1000)

    }
    best_model = None
    best_accuracy = 0
    best_model_name = None
    accuracies = []
    for name, model in models.items(): # run all model to find best accuracy for best model
        model.fit(X_train_scaled, y_train)
        predictions = model.predict(X_test_scaled)
        accuracy = accuracy_score(y_test, predictions)
        accuracies.append(accuracy)
        if accuracy > best_accuracy: # select best model based on higher accuracy
            best_accuracy = accuracy
            best_model = model
            best_model_name = name
    st.write(f"Best Model: {best_model_name} with Accuracy: {best_accuracy:.2f}") # show model
    st.bar_chart(pd.DataFrame(accuracies, index=models.keys(), columns=['Accuracy'])) #show accuracy
    predictions = best_model.predict(X_test_scaled)
### EVALUATION FUNCTION
st.header("**Evaluation Metrics 📝**",divider='red')
if selected_model in ['🎖️ Clustering', '🎖️ Classification']:
    st.write(f"<span style='color:#FFA500; font-size:30px;'>Accuracy 🎯</span>: {accuracy_score(y_test, predictions):.2f}", unsafe_allow_html=True)
    st.write('---')
    st.write(f"<span style='color:#FFA500; font-size:30px;'>F1-Score 💢</span>: {f1_score(y_test, predictions, average='weighted'):.2f}", unsafe_allow_html=True)
    report = classification_report(y_test, predictions, output_dict=True) # make classification report
    report_df = pd.DataFrame(report).transpose() # covert to data frame
    st.write('---')
    st.write("<span style='color:#FFA500; font-size:30px;'>Classification Report (DataFrame Format) 📄</span>:", unsafe_allow_html=True)
    st.dataframe(report_df) #print report
    conf_matrix = confusion_matrix(y_test, predictions) #make confusion matrix
    st.write('---')
    st.write("<span style='color:#FFA500; font-size:30px;'>confussion matrix (DataFrame Format) ❌</span>:", unsafe_allow_html=True)
    fig, ax = plt.subplots() #plot matrix
    sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    st.pyplot(fig) #show plot 
elif selected_model ==  '🎖️ Regression':
    st.write(f"<span style='color:#FFA500; font-size:30px;'>R² Score 📄</span>: {r2_score(y_test, predictions):.2f}",unsafe_allow_html=True)
    st.write('---')
    st.write(f"<span style='color:#FFA500; font-size:30px;'>Mean Squared Error (MSE) 💢</span>: {mean_squared_error(y_test, predictions):.2f}",unsafe_allow_html=True)
    st.write('---')
    st.write(f"<span style='color:#FFA500; font-size:30px;'>Mean Absolute Error (MAE) 💢</span>: {mean_absolute_error(y_test, predictions):.2f}",unsafe_allow_html=True)
    corr = np.corrcoef(y_test, predictions)[0, 1]  #find correlation
    st.write('---')
    st.write(f"<span style='color:#FFA500; font-size:30px;'>Correlation ❌</span>: {corr:.2f}",unsafe_allow_html=True)
    st.write('---')
    fig, ax = plt.subplots() #print graph
    sns.scatterplot(x=y_test, y=predictions, ax=ax)
    ax.set_xlabel("Actual")
    ax.set_ylabel("Predicted")
    ax.set_title("Actual vs Predicted")
    st.pyplot(fig) #print plot
st.write('---')
### ANALYSIS
st.header("Correlation Analysis with Heatmap 📝",divider='red')
data = generate_dummy_data() # get data as array
st.write("<span style='color:#008000; font-size:30px;'>Dataset preview ✉️</span>:", unsafe_allow_html=True)
st.dataframe(data.head()) # convert to dataframe
correlation_matrix = data.corr() #find corelation
st.write('---')
st.write("<span style='color:#008000; font-size:30px;'>Correlation Matrix 📑</span>:", unsafe_allow_html=True)
st.dataframe(correlation_matrix) #convert to data frame
st.write('---')
st.write("<span style='color:#008000; font-size:30px;'>Correlation Heatmap 🔥</span>:", unsafe_allow_html=True)
plt.figure(figsize=(10, 6))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0) # create heat map
st.pyplot(plt)
st.write('---')
fig = px.scatter(x=y_test, y=predictions, labels={'x': 'Actual', 'y': 'Predicted'}, title="Actual vs Predicted")
st.plotly_chart(fig)         #print plot