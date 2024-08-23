import pandas as pd
import joblib
import matplotlib.pyplot as plt
import streamlit as st
import statistics
import os
from langchain_groq import ChatGroq
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import HumanMessage,SystemMessage
from dotenv import load_dotenv
load_dotenv()


def model_summary(modelname, data_positive, data_negative):
    os.environ["GROQ_API_KEY"] = os.getenv("groq_key")
    model = ChatGroq(model=modelname)
    data_positive = data_positive[:20]
    data_negative = data_negative[:20]
    st.write(f"Model name is ---> {modelname}")
    messege = [
        SystemMessage(content="Act like an ai system whose job to give pros based on user review.Dont write any user comments just give 3 points."),
        HumanMessage(content=f"positive comments-->{data_positive}")
    ]
    result_positive  = model.invoke(messege)
    parser = StrOutputParser()
    response_postitive = parser.invoke(result_positive)
    messege = [
        SystemMessage(content="Act like an ai system whose job to give cons based on user review. Dont write any user comments just give 3 points."),
        HumanMessage(content=f"negative comments-->{data_negative}")
    ]
    result_negative = model.invoke(messege)
    parser = StrOutputParser()
    response_negative = parser.invoke(result_negative)

    return response_postitive, response_negative
def batch_prediction(data):
    loaded_model = joblib.load('xgboost_model.pkl')

    # Step 2: Load the saved TF-IDF vectorizer
    loaded_vectorizer = joblib.load('tfidf_vectorizer.pkl')
    new_data_tfidf = loaded_vectorizer.transform(data)

    # Step 4: Make predictions using the loaded model
    predictions = loaded_model.predict(new_data_tfidf)

    # Optional: Convert numerical predictions back to original labels
    predicted_labels = ["Computer Generated" if pred == 1 else "Original Review" for pred in predictions]
    Computer_generatedConfidance = []
    prediction_probs = loaded_model.predict_proba(new_data_tfidf)
    for i in prediction_probs:
        Computer_generatedConfidance.append(i[1])
    return predicted_labels, Computer_generatedConfidance



# Title
st.title("CSV File Upload")

# Upload CSV File
uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

if uploaded_file is not None:
    # Display the file content as a dataframe
    df = pd.read_csv(uploaded_file)
    st.write("Uploaded CSV file:")
    st.write(df)
    df = df.dropna()
    data = df['comment'].tolist()
    prediction, Computer_generatedConfidance = batch_prediction(data=data)
    df["prediction"] = prediction
    df["Computer_generatedConfidance"] = Computer_generatedConfidance
    # Scan button
    if st.button("Analysis"):
        # Processing code here
        st.write("Scanning the uploaded CSV file...")


        st.write(df)
        label_counts = df['prediction'].value_counts()

        # Plot the bar chart
        st.write("Bar Plot of 'prediction' Column:")

        fig, ax = plt.subplots()
        label_counts.plot(kind='bar', ax=ax)
        ax.set_xlabel('prediction')
        ax.set_ylabel('Count')
        ax.set_title('Bar Plot of Labels')
        st.pyplot(fig)

        rating_original = []
        for i in df["rating"]:
            rating_original.append(int(i[0]))
        mean_rating = statistics.mean(rating_original)
        st.write(f"Amazon rating ---> {mean_rating}")
        df_original = df[df["prediction"]=="Original Review"]

        # st.write(df_original)
        rating_updated = []
        for i in df_original["rating"]:
            rating_updated.append(int(i[0]))


        mean_rating_updated = statistics.mean(rating_updated)

        st.write(f"Amazon Adjusted rating ---> {mean_rating_updated}")



    models = [
        "Gemma2-9b-it",
        "Gemma-7b-it",
        "Llama-3.1-70b-versatile",
        "Llama-3.1-8b-instant",
        "Llama3-70b-8192",
        "Llama3-8b-8192",
        "Mixtral-8x7b-32768"
    ]
    selected_model = st.selectbox("Select a model:", models)

    if st.button("summary"):
        # Function that generates a summary based on the selected model
        print(df.columns)
        df_original = df[df["prediction"] == "Original Review"]
        data_positive = []
        data_negative = []
        for index, row in df_original.iterrows():
            comment = row['comment']
            rating = row['rating']
            if rating[0]=="1":
                data_negative.append(comment)
            if rating[0]=="5":
                data_positive.append(comment)
        print(len(data_positive))
        print("---------------------------------------")
        print(len(data_negative))
        response_postitive, response_negative = model_summary(modelname=selected_model, data_positive = data_positive, data_negative = data_negative )

        # Display the model response
        st.write(f"Model response ---> {response_postitive}")
        st.write(f"Model response ---> {response_negative}")

else:
    st.info("Please upload a CSV file to proceed.")






