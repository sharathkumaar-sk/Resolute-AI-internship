import pandas as pd
import streamlit as st
from classification import train_and_predict
from clustering import extract_and_predict

# Load the raw data
df = pd.read_excel('Dataset/rawdata.xlsx')
train_path = 'Dataset/train.xlsx'
test_path = 'Dataset/test.xlsx'

# Convert 'date' and 'time' columns to strings
df['date'] = df['date'].astype(str)
df['time'] = df['time'].astype(str)

# Combine 'date' and 'time' columns into a single datetime column
df['date time'] = pd.to_datetime(df['date'] + ' ' + df['time'], errors='coerce')


# Calculate the datewise total duration for each inside and outside
inside_duration_df = df[df['position'] == 'inside'].groupby('date')['position'].count().reset_index(name='inside_duration')
outside_duration_df = df[df['position'] == 'outside'].groupby('date')['position'].count().reset_index(name='outside_duration')

duration_df = pd.merge(inside_duration_df, outside_duration_df, on='date', how='outer').fillna(0)

# Calculate the datewise number of picking and placing activity done
picking_count_df = df[df['activity'] == 'picked'].groupby('date')['activity'].count().reset_index(name='picking_count')
placing_count_df = df[df['activity'] == 'placed'].groupby('date')['activity'].count().reset_index(name='placing_count')

activity_count_df = pd.merge(picking_count_df, placing_count_df, on='date', how='outer').fillna(0)

# Call the function and get the results
val_acc, train_acc, test_predictions = train_and_predict(train_path, test_path)

result = extract_and_predict()
test_predictions_str = ' '.join(test_predictions)

# Create a Streamlit app with three columns
st.title("Activity Analysis Classification and Clustering")

st.header("Datewise Analysis")
st.write("Datewise Inside and Outside Count")
st.write(duration_df)
st.write("Datewise Activity Count")
st.write(activity_count_df)


st.header("Classification")
st.write("Validation Accuracy",val_acc)
st.write("Training Accuracy",train_acc)
st.write("Predicted Target Values for Test Dataset: ",test_predictions)

st.header("Clustering")
st.write("Cluster Label:",result['cluster_label'])
st.write("Most Significant Feature: ", result["most_significant_feature"])
st.write("New Data Value: ", result["new_data_value"])
st.write("Centroid Value: ", result["centroid_value"])
