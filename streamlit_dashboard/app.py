import os, sys
import json
import re
import glob
import numpy as np
import pandas as pd
import streamlit as st
from gensim import corpora
from itertools import cycle
from datetime import datetime

import matplotlib.pyplot as plt
from datetime import datetime
from collections import defaultdict
from gensim.models import LdaModel
from collections import defaultdict
from matplotlib import pyplot as plt
from gensim.utils import simple_preprocess
from gensim.models.coherencemodel import CoherenceModel
from gensim.parsing.preprocessing import preprocess_string
sys.path.append('./')

from src.loader import SlackDataLoader
from src.utils import get_messages_reply_timestamp_from_channel,get_channel_messages_replies_timestamp,get_all_events_timestamp_on_channel,get_messages_from_channel,from_msg_get_replies_text, get_messages_timestamp_from_channel, from_msg_get_replies_text_with_specific_date


st.set_page_config(
    page_title="Slack Message Analysis",
    page_icon=":chart_with_upwards_trend:",
    layout="wide",
)

st.sidebar.title("Navigation")
selected_page = st.sidebar.radio(
    "Go to", ["Time Differences", "Topic Modeling", "Messages by Day", "Top 10 or bottom 10 users", "Top Messages","Channel Activity","Messages difference"]
)

st.title("Slack Message Analysis Dashboard")

with st.container():

    if selected_page == "Time Differences":

        st.header("Time Differences Analysis")

        # Distribution difference between consecutive messages in each channel
        st.subheader("Distribution of Time Differences Between Consecutive Messages")
        st.text("Channel: all-week1")

        # Your histogram plot here
        data_loader = SlackDataLoader(path='./dataset/')
        channels_data = data_loader.get_channels()

        path_channel = f"./dataset/all-week1/"
        time_differences = []

        stamps, _ = get_messages_reply_timestamp_from_channel(path_channel)
        for i in range(1, len(stamps)):
            prev_msg_timestamp = float(stamps[i-1][0])
            curr_msg_timestamp = float(stamps[i][0])
            time_difference = curr_msg_timestamp - prev_msg_timestamp
            time_differences.append(time_difference)

        fig, ax = plt.subplots()
        ax.hist(time_differences, bins=20, edgecolor='blue')
        ax.set_xlabel('Time Difference (seconds)')
        ax.set_ylabel('Frequency')
        ax.set_title(f'Distribution of Time Differences Between Consecutive Messages\nChannel: all-week1')
        st.pyplot(fig)
        plt.close()

        # Distribution difference between consecutive replies in each channels
        st.subheader("Distribution of Time Differences Between Consecutive Replies")
        st.text("Channel: all-week1")

        # Your histogram plot here
        path_channel = f"./dataset/all-week1"
        reply_timestamps = get_channel_messages_replies_timestamp(path_channel)

        datetime_timestamps = [datetime.fromtimestamp(float(timestamp)) for timestamp in reply_timestamps]
        time_differences = np.diff(datetime_timestamps)
        time_differences_seconds = [td.total_seconds() for td in time_differences]

        fig, ax = plt.subplots()
        ax.hist(time_differences_seconds, bins=20, edgecolor='blue')
        ax.set_xlabel('Time Difference (seconds)')
        ax.set_ylabel('Frequency')
        ax.set_title(f'Distribution of Time Differences Between Consecutive Replies\nChannel: all-week1')
        st.pyplot(fig)
        plt.close()

        # Distribution difference between consecutive events in each channels
        st.subheader("Distribution of Time Differences Between Consecutive Events")
        st.text("Channel: all-week1")

        # Your histogram plot here
        path_channel = f"./dataset/all-week1"
        event_timestamps = get_all_events_timestamp_on_channel(path_channel)
        datetime_timestamps = [datetime.fromtimestamp(float(timestamp)) for timestamp in event_timestamps]

        time_differences = np.diff(datetime_timestamps)
        time_differences_seconds = [td.total_seconds() for td in time_differences]

        fig, ax = plt.subplots()
        ax.hist(time_differences_seconds, bins=20, edgecolor='blue')
        ax.set_xlabel('Time Difference (seconds)')
        ax.set_ylabel('Frequency')
        ax.set_title(f'Distribution of Time Differences Between Consecutive Events\nChannel: all-week1')
        st.pyplot(fig)
        plt.close()

    elif selected_page == "Topic Modeling":

        st.header("Topic Modeling")
        data_loader = SlackDataLoader(path='./dataset/')
        channels_data = data_loader.get_channels()
    
        def get_total_messages_and_replies(channels_data):
            '''
            Get the total messages and replies for each channel
            '''
            all_messages = []
            all_replies = []

            
            channel_path = f"./dataset/all-week1"
            channel_messages = get_messages_from_channel(channel_path)

            # Extract replies for each message
            channel_messages["replies"] = channel_messages.apply(lambda msg: from_msg_get_replies_text(msg), axis=1)

            # Append messages and replies to respective lists
            all_messages.extend(channel_messages["text"].tolist())
            all_replies.extend(channel_messages["replies"].tolist())

            return all_messages, all_replies




        # Assuming you have a function 'get_total_messages_and_replies' defined somewhere
        all_messages, all_replies = get_total_messages_and_replies(channels_data)

        # Define a function to preprocess text
        def preprocess_text(text):
            return simple_preprocess(text, deacc=True, min_len=2)

        # Define a function to preprocess messages
        def preprocess_messages(messages):
            return [preprocess_text(message) for message in messages]

        # Preprocess the messages
        processed_messages = preprocess_messages(all_messages)
        dictionary = corpora.Dictionary(processed_messages)
        corpus = [dictionary.doc2bow(message) for message in processed_messages]

        num_topics = 1
        lda_model = LdaModel(corpus, num_topics=num_topics, id2word=dictionary, passes=10)

        topics = lda_model.show_topics(num_topics=num_topics, num_words=10, formatted=False)

        # Display topics in Streamlit
        st.title("LDA Topics")

        for topic_id, topic in topics:
            st.subheader(f"Topic {topic_id + 1}:")
            for word, prob in topic:
                st.write(f"- {word} (Probability: {prob:.4f})")

    elif selected_page == "Messages by Day":

        st.header("Messages Grouped by Day")

        #All_messages in one_day
        all_messages_from_channel = defaultdict(list)
        path_channel = f"./dataset/all-week1"
        all_messages = get_messages_timestamp_from_channel(path_channel)
        for date, messages in all_messages.items():
            all_messages_from_channel[date[:10]].extend(messages)

        for date, list_of_messages in list(all_messages_from_channel.items())[:10]:
            formatted_date = datetime.fromtimestamp(int(date)).strftime("%B %d, %Y %H:%M:%S")
            st.subheader(f"Date: {formatted_date}")
            
            for message in list_of_messages[:10]:
                st.write(message)
            
            st.write("\n")

        st.subheader("Concatenated Messages and Replies Grouped by Day")
        all_messages_and_replies_from_channels = defaultdict(list)
        all_messages = get_messages_timestamp_from_channel(path_channel)
        all_replies = from_msg_get_replies_text_with_specific_date(path_channel)

        for date, messages in all_messages.items():
            date_obj = datetime.fromtimestamp(float(date))
            all_messages_and_replies_from_channels[date_obj.strftime("%Y-%m-%d")].extend(messages)

        for date, replies in all_replies.items():
            date_obj = datetime.fromtimestamp(float(date))
            all_messages_and_replies_from_channels[date_obj.strftime("%Y-%m-%d")].extend(replies)

        st.title("Concatenated Messages and Replies Grouped by Day")

        for date, list_of_messages_and_replies in list(all_messages_and_replies_from_channels.items())[:10]:
            if isinstance(date, datetime): 
                formatted_date = date.strftime("%B %d, %Y %H:%M:%S")
            else:
                formatted_date = date 
            st.subheader(f"Date: {formatted_date}")
            
            for message in list_of_messages_and_replies[:3]:
                st.write(message)
            
            st.write("\n")


    elif selected_page == "Top 10 or bottom 10 users":

        def get_user_reply_counts(path):
            user_reply_counts = defaultdict(int)

            # Get the user map
            for json_file in glob.glob(f"{path}*.json"):
                with open(json_file, 'r') as slack_data:
                    try:
                        data = json.load(slack_data)
                    except json.JSONDecodeError:
                        continue

                    if not isinstance(data, list):
                        continue

                    for message in data:
                        if not isinstance(message, dict):
                            continue

                        user_id = message.get('user')

                        if user_id is not None:
                            user_reply_counts[user_id] += message.get('reply_count', 0)

            return user_reply_counts

        # Set paths
        slack_data_path = "./dataset/"
        path = "./dataset/all-week1/"

        # Load Slack data
        slack_loader = SlackDataLoader(slack_data_path)
        user_names_by_id, user_ids_by_name = slack_loader.get_user_map()

        user_reply_counts = get_user_reply_counts(path)
        sorted_users = sorted(user_reply_counts.items(), key=lambda x: x[1], reverse=True)
        st.title("Top and Bottom Users by Reply Count")

        # Top 10 Users
        st.subheader("Top 10 Users by Reply Count:")
        for user_id, count in sorted_users[:10]:
            st.write(f"{user_names_by_id[user_id]} has {count} replies")

        # Bottom 10 Users
        st.subheader("Bottom 10 Users by Reply Count:")
        for user_id, count in sorted_users[-10:]:
            st.write(f"{user_names_by_id[user_id]} has {count} replies")

        # Function to get user message counts
        def get_user_message_counts(path):
            user_message_counts = defaultdict(int)

            for json_file in glob.glob(f"{path}*.json"):
                with open(json_file, 'r') as slack_data:
                    try:
                        data = json.load(slack_data)
                    except json.JSONDecodeError:
                        continue

                    if not isinstance(data, list):
                        continue

                    for message in data:
                        if not isinstance(message, dict):
                            continue

                        user_id = message.get('user')

                        if user_id is not None:
                            user_message_counts[user_id] += 1
            return user_message_counts

        # Set paths
        slack_data_path = "./dataset/"
        path = "./dataset/all-week1/"

        # Load Slack data
        slack_loader = SlackDataLoader(slack_data_path)
        user_names_by_id, user_ids_by_name = slack_loader.get_user_map()
        user_message_counts = get_user_message_counts(path)
        sorted_users = sorted(user_message_counts.items(), key=lambda x: x[1], reverse=True)
        st.title("Top and Bottom Users by Message Count")

        # Top 10 Users
        st.subheader("Top 10 Users by Message Count:")
        for user_id, count in sorted_users[:10]:
            st.write(f"{user_names_by_id[user_id]} has {count} messages")

        # Bottom 10 Users
        st.subheader("Bottom 10 Users by Message Count:")
        for user_id, count in sorted_users[-10:]:
            st.write(f"{user_names_by_id[user_id]} has {count} messages")

        # Function to get user reaction counts
        def get_user_reaction_counts(path):
            user_reaction_counts = defaultdict(int)

            for json_file in glob.glob(f"{path}*.json"):
                with open(json_file, 'r') as slack_data:
                    try:
                        data = json.load(slack_data)
                    except json.JSONDecodeError:
                        continue

                    if not isinstance(data, list):
                        continue

                    for message in data:
                        if not isinstance(message, dict):
                            continue

                        user_id = message.get('user')
                        reactions = message.get('reactions', [])

                        if user_id is not None:
                            user_reaction_counts[user_id] += len(reactions)

            return user_reaction_counts

        # Set paths
        slack_data_path = "./dataset/"
        path = "./dataset/all-week1/"

        # Load Slack data
        slack_loader = SlackDataLoader(slack_data_path)
        user_names_by_id, user_ids_by_name = slack_loader.get_user_map()
        user_reaction_counts = get_user_reaction_counts(path)
        sorted_users = sorted(user_reaction_counts.items(), key=lambda x: x[1], reverse=True)
        st.title("Top and Bottom Users by Reaction Count")

        # Top 10 Users
        st.subheader("Top 10 Users by Reaction Count:")
        for user_id, count in sorted_users[:10]:
            st.write(f"{user_names_by_id[user_id]} has {count} reactions")

        # Bottom 10 Users
        st.subheader("Bottom 10 Users by Reaction Count:")
        for user_id, count in sorted_users[-10:]:
            st.write(f"{user_names_by_id[user_id]} has {count} reactions")

    elif selected_page == "Top Messages":
       
        # Function to get top messages by replies
        def get_top_messages_by_replies(path, top_n=10):
            message_replies = defaultdict(int)

            for json_file in glob.glob(f"{path}*.json"):
                with open(json_file, 'r') as slack_data:
                    try:
                        data = json.load(slack_data)
                    except json.JSONDecodeError:
                        continue

                    if not isinstance(data, list):
                        continue

                    for message in data:
                        if not isinstance(message, dict):
                            continue

                        reply_count = message.get('reply_count', 0)
                        message_text = message.get('text', '')
                        message_replies[message_text] += reply_count

            sorted_messages = sorted(message_replies.items(), key=lambda x: x[1], reverse=True)
            top_messages = sorted_messages[:top_n]

            return top_messages

        # Set paths
        slack_data_path = "./dataset/"
        path = "./dataset/all-week1/"

        # Load Slack data
        slack_loader = SlackDataLoader(slack_data_path)
        top_messages = get_top_messages_by_replies(path)
        st.title("Top Messages by Replies")

        for message, reply_count in top_messages:
            st.subheader("Message:")
            st.write(message)
            st.subheader(f"Replies: {reply_count}")
            st.write("\n")

    elif selected_page == "Channel Activity":

        # Function to calculate channel activity based on members
        def calculate_channel_activity(channels_data):
            channel_activity = {}
            for channel in channels_data:
                channel_name = channel['name']
                channel_id = channel['id']
                members_count = len(channel['members'])
                channel_activity[channel_name] = members_count

            return channel_activity

        # Set paths
        slack_data_path = "./dataset/"
        data_loader = SlackDataLoader(path=slack_data_path)
        channels_data = data_loader.get_channels()

        channel_activity_data = calculate_channel_activity(channels_data)
        highest_activity_channel = max(channel_activity_data, key=channel_activity_data.get)

        st.title("Channel with the Highest Activity")
        st.subheader(f"The channel with the highest activity is: {highest_activity_channel}")
        x_values = list(channel_activity_data.values())
        y_values = [0] * len(x_values)
        channel_names = list(channel_activity_data.keys())

        # Create scatter plot
        fig, ax = plt.subplots()
        scatter = ax.scatter(x_values, y_values, c=range(len(channel_names)), cmap='viridis')
        ax.set_xlabel('Number of Members')
        ax.set_ylabel('Placeholder for Total Replies and Reactions')
        ax.set_title('Channel Activity (Number of Members)')
        fig.colorbar(scatter, label='Channels')
        st.pyplot(fig)

        # Function to calculate reply and reaction counts for each channel
        def calculate_channel_counts(channels_data):
            x_axis_message_count = {}
            y_axis_reply_reaction_count = {}

            for channel in channels_data:
                path = channel['name']
                replay_df = data_loader.slack_parser(f"./dataset/{channel['name']}")
                reaction_df = data_loader.parse_slack_reaction(f"./dataset/{channel['name']}", channel['name'])

                total_reply_count = replay_df['reply_count'].sum()
                total_reaction_count = reaction_df['reaction_count'].sum()

                x_axis_message_count[channel['name']] = len(replay_df)
                y_axis_reply_reaction_count[channel['name']] = total_reply_count + total_reaction_count

            return x_axis_message_count, y_axis_reply_reaction_count

        # Set paths
        slack_data_path = "./dataset/"
        data_loader = SlackDataLoader(path=slack_data_path)
        channels_data = data_loader.get_channels()
        x_axis_message_count, y_axis_reply_reaction_count = calculate_channel_counts(channels_data)
        st.title("Channel Activity Analysis")
        st.subheader("2D Scatter Plot: Messages vs. Replies + Reactions")

        # Scatter plot setup
        channels = list(x_axis_message_count.keys())
        x_values = list(x_axis_message_count.values())
        y_values = list(y_axis_reply_reaction_count.values())

        # Create scatter plot
        fig, ax = plt.subplots(figsize=(10, 6))
        scatter = ax.scatter(x_values, y_values, c=range(len(channels)), cmap='viridis', edgecolors='black', linewidths=0.5)
        ax.set_xlabel('Number of Messages')
        ax.set_ylabel('Sum of Replies and Reactions')
        ax.set_title('Channel Activity: Messages vs. Replies + Reactions')

        # Add channel labels
        for i, channel in enumerate(channels):
            ax.annotate(channel, (x_values[i], y_values[i]), fontsize=8, ha='right')

        cbar = fig.colorbar(scatter, ax=ax, label='Channels')
        cbar.outline.set_edgecolor('black')
        cbar.outline.set_linewidth(0.5)
        cbar.ax.tick_params(axis='y', color='black', width=0.5)
        ax.set_facecolor('#f0f0f0')
        st.pyplot(fig)

    elif selected_page == "Messages difference":

        # Function to calculate time difference in seconds
        def calculate_time_difference_seconds(start_time, end_time):
            start_datetime = datetime.utcfromtimestamp(float(start_time))
            end_datetime = datetime.utcfromtimestamp(float(end_time))
            time_difference_seconds = (end_datetime - start_datetime).total_seconds()
            return time_difference_seconds

        # Function to process channels data and generate the scatter plot
        def process_channels_data(channels_data):
            channels_stamps = {}
            channels_no_reply_count = {}
            message_reply_less_than_5_minutes = 0
            total_messages = 0

            for channel in channels_data:
                path_channel = (f"./dataset/{channel['name']}")
                stamps, no_reply_messages_count = get_messages_reply_timestamp_from_channel(path_channel)
                channels_stamps[channel["name"]] = stamps
                channels_no_reply_count[channel["name"]] = no_reply_messages_count

                for msg_timestamp, reply_timestamp in stamps:
                    time_difference = calculate_time_difference_seconds(msg_timestamp, reply_timestamp)
                    if time_difference <= 300:  # 300 seconds = 5 minutes
                        message_reply_less_than_5_minutes += 1

                total_messages += (len(stamps) + no_reply_messages_count)

            # Display results
            st.title("Message Reply Analysis")
            st.write("Total number of messages:", total_messages)
            st.write("Number messages replied within 5 minutes:", message_reply_less_than_5_minutes)
            st.write("The Fraction of messages replied within 5 minutes:", message_reply_less_than_5_minutes / total_messages)

            # Scatter plot setup
            fig, ax = plt.subplots(figsize=(10, 6))
            colors = cycle(['#FF5733', '#33FF57', '#5733FF', '#FF33E6', '#FFD133'])  # More colorful palette

            for channel_name, message_timestamps in channels_stamps.items():
                x_values = [calculate_time_difference_seconds(msg_timestamp, reply_timestamp) for msg_timestamp, reply_timestamp in message_timestamps]
                y_values = [datetime.utcfromtimestamp(float(msg_timestamp)).hour for msg_timestamp, _ in message_timestamps]

                current_color = next(colors)
                ax.scatter(x_values, y_values, label=channel_name, alpha=0.5, color=current_color)

            # Set plot labels and title
            ax.set_xlabel('Time Difference (seconds)')
            ax.set_ylabel('Time of the Day (hour)')
            ax.set_title('Scatter Plot of Time Difference vs. Time of the Day')

            # Set background color
            ax.set_facecolor('#F0F0F0')  # Light gray background
            fig.set_facecolor('#E0E0E0')  # Slightly darker background

            # Move the legend outside and add more spacing
            ax.legend(loc='upper left', bbox_to_anchor=(1, 1), title='Channels', frameon=False)
            plt.subplots_adjust(right=0.7) 
            st.pyplot(fig)

        # Main function
        if __name__ == "__main__":
            data_loader = SlackDataLoader(path='./dataset/')
            channels_data = data_loader.get_channels()
            process_channels_data(channels_data)





















































 
























