import streamlit as st
import os, sys
from matplotlib import pyplot as plt
from nltk.corpus import stopwords
from collections import defaultdict
from datetime import datetime
from itertools import cycle
import numpy as np
import sys
sys.path.append('./')
# Import your custom modules or functions
from src.loader import SlackDataLoader
from src.utils import get_messages_reply_timestamp_from_channel

# Set page configuration
st.set_page_config(page_title="Slack Message Analysis")

# Add content to the Streamlit app
with st.container():
    st.subheader("Hey")
    st.title("A")
    st.write("B")

    # Add a selectbox to the sidebar:
    add_selectbox = st.sidebar.selectbox(
        'How would you like to be contacted?',
        ('Email', 'Home phone', 'Mobile phone')
    )

    # Add a slider to the sidebar:
    add_slider = st.sidebar.slider(
        'Select a range of values',
        0.0, 100.0, (25.0, 75.0)
    )

    # Distribution difference between consecutive messages in each channel
    data_loader = SlackDataLoader(path='./dataset/')
    channels_data = data_loader.get_channels()

    # for channel in channels_data:
    path_channel = f"./dataset/all-week1/"
    time_differences = []

    stamps, _ = get_messages_reply_timestamp_from_channel(path_channel)

    # Calculate time differences between consecutive messages
    for i in range(1, len(stamps)):
        prev_msg_timestamp = float(stamps[i-1][0])
        curr_msg_timestamp = float(stamps[i][0])
        time_difference = curr_msg_timestamp - prev_msg_timestamp
        time_differences.append(time_difference)

    # Plot histogram of time differences for the current channel
    # Plot histogram of time differences for the current channel
    fig, ax = plt.subplots()
    ax.hist(time_differences, bins=20, edgecolor='blue')
    ax.set_xlabel('Time Difference (seconds)')
    ax.set_ylabel('Frequency')
    ax.set_title(f'Distribution of Time Differences Between Consecutive Messages\nChannel: all-week1')

    # Display the Matplotlib figure in Streamlit
    st.pyplot(fig)

    # Close the Matplotlib figure to avoid the warning
    plt.close()

