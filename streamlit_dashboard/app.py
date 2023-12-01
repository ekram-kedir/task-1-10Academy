import streamlit as st
import os, sys
from matplotlib import pyplot as plt
from nltk.corpus import stopwords
from collections import defaultdict
from datetime import datetime
from itertools import cycle
import numpy as np
import streamlit as st
import json
import pandas as pd
import re
from collections import defaultdict
from datetime import datetime
import numpy as np

from gensim import corpora
from gensim.models import LdaModel
from gensim.models.coherencemodel import CoherenceModel
from gensim.utils import simple_preprocess
from gensim.parsing.preprocessing import preprocess_string
sys.path.append('./')
# Import your custom modules or functions
from src.loader import SlackDataLoader
from src.utils import get_messages_reply_timestamp_from_channel,get_channel_messages_replies_timestamp,get_all_events_timestamp_on_channel,get_messages_from_channel,from_msg_get_replies_text,get_messages_timestamp_from_channel, from_msg_get_replies_text_with_specific_date

# Set page configuration
st.set_page_config(page_title="Slack Message Analysis")

# Add content to the Streamlit app
with st.container():
    st.subheader("Hey")
    st.title("A")
    st.write("B")

# Distribution difference between consecutive messages in each channel
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





#Topic Modeling 

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

# Assuming you have already loaded 'all_messages' using the appropriate function

# Preprocess the messages
processed_messages = preprocess_messages(all_messages)

# Create dictionary and corpus
dictionary = corpora.Dictionary(processed_messages)
corpus = [dictionary.doc2bow(message) for message in processed_messages]

# Train the LDA model
num_topics = 1
lda_model = LdaModel(corpus, num_topics=num_topics, id2word=dictionary, passes=10)

# Get the top 10 words for each topic
topics = lda_model.show_topics(num_topics=num_topics, num_words=10, formatted=False)

# Display topics in Streamlit
st.title("LDA Topics")

for topic_id, topic in topics:
    st.subheader(f"Topic {topic_id + 1}:")
    for word, prob in topic:
        st.write(f"- {word} (Probability: {prob:.4f})")
    st.write("\n")


# import matplotlib.pyplot as plt

# dates = list(positive_sentiments.keys()) 

# positive_counts = list(positive_sentiments.values())
# negative_counts = list(negative_sentiments.values())
# neutral_counts = list(neutral_sentiments.values())

# plt.figure(figsize=(10, 6))
# plt.plot(dates, positive_counts, label='Positive Sentiments')
# plt.plot(dates, negative_counts, label='Negative Sentiments')
# plt.plot(dates, neutral_counts, label='Neutral Sentiments')

# plt.xlabel('Date')
# plt.ylabel('Sentiment Counts')
# plt.title('Time Series Trend of Sentiments')
# plt.xticks(rotation=45)
# plt.legend()

# plt.tight_layout()
# plt.show()






# #Perform sentimental analysis

# def perform_sentiment_analysis(messages):
#     sia = SentimentIntensityAnalyzer()
#     positive_sentiments = 0
#     negative_sentiments = 0
#     neutral_sentiments = 0
    
#     for message in messages:
#         sentiment_scores = sia.polarity_scores(message)
#         sentiment_score = sentiment_scores['compound']
        
#         if sentiment_score > 0:
#             positive_sentiments += 1
#         elif sentiment_score < 0:
#             negative_sentiments += 1
#         else:
#             neutral_sentiments += 1
    
#     return positive_sentiments, negative_sentiments, neutral_sentiments

# data_loader = SlackDataLoader(path='./../dataset/')
# channels_data = data_loader.get_channels()
# all_messages_and_replies_from_channels = defaultdict(list)

# for channel in channels_data:
#     path_channel = f"../dataset/{channel['name']}"
#     all_messages = get_messages_timestamp_from_channel(path_channel)
#     all_replies = from_msg_get_replies_text_with_specific_date(path_channel)
    
#     for date, messages in all_messages.items():
#         all_messages_and_replies_from_channels[date[:10]].extend(messages)
#     for date, replies in all_replies.items():
#         all_messages_and_replies_from_channels[date[:10]].extend(replies)

# positive_sentiments = {}
# negative_sentiments = {}
# neutral_sentiments = {}

# for date, messages in all_messages_and_replies_from_channels.items():
#     positive_count, negative_count, neutral_count = perform_sentiment_analysis(messages)
#     positive_sentiments[date] = positive_count
#     negative_sentiments[date] = negative_count
#     neutral_sentiments[date] = neutral_count

# print("Positive Sentiments:")
# for date, count in positive_sentiments.items():
#     date = datetime.datetime.fromtimestamp(int(date))
#     formatted_date = date.strftime("%B %d, %Y %H:%M:%S")
#     print(formatted_date, count)

# print("Negative Sentiments:")
# for date, count in negative_sentiments.items():
#     date = datetime.datetime.fromtimestamp(int(date))
#     formatted_date = date.strftime("%B %d, %Y %H:%M:%S")
#     print(formatted_date, count)

# print("Neutral Sentiments:")
# for date, count in neutral_sentiments.items():
#     date = datetime.datetime.fromtimestamp(int(date))
#     formatted_date = date.strftime("%B %d, %Y %H:%M:%S")
#     print(formatted_date, count)






# #Concatenate all messages and replies in the same day as one big text

# from collections import defaultdict
# data_loader = SlackDataLoader(path='./../dataset/')
# channels_data = data_loader.get_channels()
# all_messages_and_replies_from_channels = defaultdict(list)

# for channel in channels_data:

#     path_channel = f"../dataset/{channel['name']}"
#     all_messages = get_messages_timestamp_from_channel(path_channel)
#     all_replies = from_msg_get_replies_text_with_specific_date(path_channel)
#     for date,messages in all_messages.items():
#         all_messages_and_replies_from_channels[date[:10]].append(messages)
#     for date,messages in all_replies.items():
#         all_messages_and_replies_from_channels[date[:10]].append(messages)

# for date, list_of_messages_and_replies in all_messages_and_replies_from_channels.items():
#     date = datetime.datetime.fromtimestamp(int(date))
#     formatted_date = date.strftime("%B %d, %Y %H:%M:%S")
#     print("Date:", formatted_date)
#     for message in list_of_messages_and_replies:
#         print(message)
#     print()





# Assuming you have a function 'get_messages_timestamp_from_channel' and 'from_msg_get_replies_text_with_specific_date' defined somewhere

# Set page configuration
st.set_page_config(page_title="Messages Grouped by Day")

all_messages_from_channel = defaultdict(list)

# Group messages by day
path_channel = f"./dataset/all-week1"
all_messages = get_messages_timestamp_from_channel(path_channel)
for date, messages in all_messages.items():
    all_messages_from_channel[date[:10]].extend(messages)

# Display messages in Streamlit
st.title("Messages Grouped by Day")

for date, list_of_messages in all_messages_from_channel.items():
    formatted_date = datetime.fromtimestamp(int(date)).strftime("%B %d, %Y %H:%M:%S")
    st.subheader(f"Date: {formatted_date}")
    for message in list_of_messages:
        st.write(message)
    st.write("\n")



















# # Top 10 and Bottom 10 users based on reply count for week1

# def get_user_reply_counts(path):
#     """Get reply counts for each user."""
#     user_reply_counts = defaultdict(int)

#     # Get the user map
#     for json_file in glob.glob(f"{path}*.json"):
#         with open(json_file, 'r') as slack_data:
#             try:
#                 data = json.load(slack_data)
#             except json.JSONDecodeError:
#                 continue

#             if not isinstance(data, list):
#                 continue

#             for message in data:
#                 if not isinstance(message, dict):
#                     continue

#                 user_id = message.get('user')

#                 if user_id is not None:
#                     user_reply_counts[user_id] += message.get('reply_count', 0)

#     return user_reply_counts

# def main():
#     slack_data_path = './../dataset/'  
#     path = './../dataset/all-week1/'
    
#     slack_loader = SlackDataLoader(slack_data_path)
#     users = slack_loader.get_users()
#     user_reply_counts = get_user_reply_counts(path)
#     user_names_by_id, user_ids_by_name = slack_loader.get_user_map()

#     sorted_users = sorted(user_reply_counts.items(), key=lambda x: x[1], reverse=True)

#     top_10_users = sorted_users[:10]
#     bottom_10_users = sorted_users[-10:]

#     print("Top 10 Users by Reply Count:")
#     for user_id, count in top_10_users:
#         print(f"{user_names_by_id[user_id]} has {count} replies")

#     print("\nBottom 10 Users by Reply Count:")
#     for user_id, count in bottom_10_users:
#         print(f"{user_names_by_id[user_id]} has {count} replies")

# if __name__ == "__main__":
#     main()




# # Top 10 and Bottom 10 users based on message count for week1

# def get_user_message_counts(path):
#     """Get message counts for each user."""
#     user_message_counts = defaultdict(int)

#     for json_file in glob.glob(f"{path}*.json"):
#         with open(json_file, 'r') as slack_data:
#             try:
#                 data = json.load(slack_data)
#             except json.JSONDecodeError:
#                 continue

#             if not isinstance(data, list):
#                 continue

#             for message in data:
#                 if not isinstance(message, dict):
#                     continue

#                 user_id = message.get('user')

#                 if user_id is not None:
#                     user_message_counts[user_id] += 1
#     return user_message_counts


# def main():

#     slack_data_path = './../dataset/' 
#     path = './../dataset/all-week1/'
#     slack_loader = SlackDataLoader(slack_data_path)
#     user_message_counts = get_user_message_counts(path)
#     users = slack_loader.get_users()
#     user_names_by_id, user_ids_by_name = slack_loader.get_user_map()

#     sorted_users = sorted(user_message_counts.items(), key=lambda x: x[1], reverse=True)

#     top_10_users = sorted_users[:10]
#     bottom_10_users = sorted_users[-10:]

#     print("Top 10 Users by Message Count:")
#     for user_id, count in top_10_users:
#         print(f"{user_names_by_id[user_id]} has {count} messages")

#     print("\nBottom 10 Users by Message Count:")
#     for user_id, count in bottom_10_users:
#         print(f"{user_names_by_id[user_id]} has {count} messages")

# if __name__ == "__main__":
#     main()


# # Top 10 and Bottom 10 users based on reaction count for week1

# def get_user_reaction_counts(path):
#     """Get reaction counts for each user."""
#     user_reaction_counts = defaultdict(int)

#     for json_file in glob.glob(f"{path}*.json"):
#         with open(json_file, 'r') as slack_data:
#             try:
#                 data = json.load(slack_data)
#             except json.JSONDecodeError:
#                 continue

#             if not isinstance(data, list):
#                 continue

#             for message in data:
#                 if not isinstance(message, dict):
#                     continue

#                 user_id = message.get('user')
#                 reactions = message.get('reactions', [])

#                 if user_id is not None:
#                     user_reaction_counts[user_id] += len(reactions)

#     return user_reaction_counts

# def main():

#     slack_data_path = './../dataset/'  
#     path = './../dataset/all-week1/'

#     slack_loader = SlackDataLoader(slack_data_path)

#     user_reaction_counts = get_user_reaction_counts(path)
#     users = slack_loader.get_users()
#     user_names_by_id, user_ids_by_name = slack_loader.get_user_map()

#     sorted_users = sorted(user_reaction_counts.items(), key=lambda x: x[1], reverse=True)

#     top_10_users = sorted_users[:10]
#     bottom_10_users = sorted_users[-10:]

#     print("Top 10 Users by Reaction Count:")
#     for user, count in top_10_users:
#         print(f"{user_names_by_id[user]} has got {count} reactions")

#     print("\nBottom 10 Users by Reaction Count:")
#     for user, count in bottom_10_users:
#         print(f"{user_names_by_id[user]}has got {count} reactions")

# if __name__ == "__main__":
#     main()






# # Top 10 messages based on replies for week1

# def get_top_messages_by_replies(path, top_n=10):
#     """Get top messages by replies."""
#     message_replies = defaultdict(int)

#     for json_file in glob.glob(f"{path}*.json"):
#         with open(json_file, 'r') as slack_data:
#             try:
#                 data = json.load(slack_data)
#             except json.JSONDecodeError:
#                 continue

#             if not isinstance(data, list):
#                 continue

#             for message in data:
#                 if not isinstance(message, dict):
#                     continue

#                 reply_count = message.get('reply_count', 0)
#                 message_text = message.get('text', '')
#                 message_replies[message_text] += reply_count

#     sorted_messages = sorted(message_replies.items(), key=lambda x: x[1], reverse=True)
#     top_messages = sorted_messages[:top_n]

#     return top_messages

# def main():

#     slack_data_path = './../dataset/'  
#     path = './../dataset/all-week1/'

#     top_messages = get_top_messages_by_replies(path)
#     print("Top 10 Messages by Replies:")
#     for message, reply_count in top_messages:
#         print(f"Message: {message}")
#         print(f"Replies: {reply_count}\n")

# if __name__ == "__main__":
#     main()



# # Top 10 messages based on reactions for week1

# def get_top_messages_by_reactions(path, top_n=10):
#     """Get top messages by reactions."""
#     message_reactions = defaultdict(int)

#     for json_file in glob.glob(f"{path}*.json"):
#         with open(json_file, 'r') as slack_data:
#             try:
#                 data = json.load(slack_data)
#             except json.JSONDecodeError:
#                 continue

#             if not isinstance(data, list):
#                 continue

#             for message in data:
#                 if not isinstance(message, dict):
#                     continue

#                 reactions = message.get('reactions', [])
#                 total_reactions = sum(reaction.get('count', 0) for reaction in reactions)
#                 message_text = message.get('text', '')
#                 message_reactions[message_text] += total_reactions

#     sorted_messages = sorted(message_reactions.items(), key=lambda x: x[1], reverse=True)
#     top_messages = sorted_messages[:top_n]

#     return top_messages

# def main():

#     path = './../dataset/all-week1/'  
#     top_messages = get_top_messages_by_reactions(path)

#     print("Top 10 Messages by Reactions:")
#     for message, reaction_count in top_messages:
#         print(f"Message: {message}")
#         print(f"Reactions: {reaction_count}\n")

# if __name__ == "__main__":
#     main()









# # Which channel has the highest activity? 
# # Which channel appears at the right top corner when you plot a 2D scatter plot where x-axis is the number of messages in the channel, y-axis is the sum of number of replies and reactions, and the color representing channels?
# #It uses members intsead of messages 

# import matplotlib.pyplot as plt

# def calculate_channel_activity(channels_data):
#     channel_activity = {}
#     for channel in channels_data:
#         channel_name = channel['name']
#         channel_id = channel['id']
#         members_count = len(channel['members'])
#         channel_activity[channel_name] = members_count

#     return channel_activity

# data_loader = SlackDataLoader(path='./../dataset/')
# channels_data = data_loader.get_channels()

# channel_activity_data = calculate_channel_activity(channels_data)
# highest_activity_channel = max(channel_activity_data, key=channel_activity_data.get)

# print(f"The channel with the highest activity is: {highest_activity_channel}")

# x_values = list(channel_activity_data.values())  
# y_values = [0] * len(x_values) 
# channel_names = list(channel_activity_data.keys())

# plt.scatter(x_values, y_values, c=range(len(channel_names)), cmap='viridis')
# plt.xlabel('Number of Members')
# plt.ylabel('Placeholder for Total Replies and Reactions')
# plt.title('Channel Activity (Number of Members)')
# plt.colorbar(label='Channels')
# plt.show()



# # Which channel has the highest activity? 
# # Which channel appears at the right top corner when you plot a 2D scatter plot where x-axis is the number of messages in the channel, y-axis is the sum of number of replies and reactions, and the color representing channels?


# x_axis_message_count = {}
# y_axis_reply_reaction_count = {}

# data_loader = SlackDataLoader(path='./../dataset/')
# channels_data = data_loader.get_channels()

# for channel in channels_data:
#     path = channel['name']
#     replay_df = data_loader.slack_parser(f"../dataset/{channel['name']}")
#     reaction_df = data_loader.parse_slack_reaction(f"../dataset/{channel['name']}", channel['name'])

#     total_reply_count = replay_df['reply_count'].sum()
#     total_reaction_count = reaction_df['reaction_count'].sum()

#     x_axis_message_count[channel['name']] = len(replay_df)
#     y_axis_reply_reaction_count[channel['name']] = total_reply_count + total_reaction_count

# channels = list(x_axis_message_count.keys())
# x_values = list(x_axis_message_count.values())
# y_values = list(y_axis_reply_reaction_count.values())

# # Start plotting using your labels
# plt.scatter(x_values, y_values, c='purple', label='Total Channels')

# # Add labels and title
# plt.xlabel('Number of Messages')
# plt.ylabel('Sum of Replies and Reactions')
# plt.title('Plot of Messages, Replies, and Reactions across Channel')

# # Add channel labels
# for i, channel in enumerate(channels):
#     plt.annotate(channel, (x_values[i], y_values[i]))

# plt.legend()
# plt.show()






# # What fraction of messages are replied within the first 5mins?
# # Plot a 2D scatter plot such that x-axis is the time difference between the message timestamp and the first reply message, y-axis is the time of the day (in 24hr format), color representing channels? 

# from datetime import datetime
# from itertools import cycle

# def calculate_time_difference_seconds(start_time, end_time):
#     """
#     Calculate the time difference in seconds between two timestamps.

#     Args:
#         start_time (float): Start timestamp.
#         end_time (float): End timestamp.

#     Returns:
#         float: Time difference in seconds.
#     """
#     start_datetime = datetime.utcfromtimestamp(float(start_time))
#     end_datetime = datetime.utcfromtimestamp(float(end_time))
#     time_difference_seconds = (end_datetime - start_datetime).total_seconds()
#     return time_difference_seconds

# def process_channels_data(channels_data):
#     channels_stamps = {}
#     channels_no_reply_count = {}
#     message_reply_less_than_5_minutes = 0
#     total_messages = 0

#     for channel in channels_data:
#         path_channel = f"../dataset/{channel['name']}"
#         stamps, no_reply_messages_count = get_messages_reply_timestamp_from_channel(path_channel)
#         channels_stamps[channel["name"]] = stamps
#         channels_no_reply_count[channel["name"]] = no_reply_messages_count

#         for msg_timestamp, reply_timestamp in stamps:
#             time_difference = calculate_time_difference_seconds(msg_timestamp, reply_timestamp)
#             if time_difference <= 300:  # 300 seconds = 5 minutes
#                 message_reply_less_than_5_minutes += 1

#         total_messages += (len(stamps) + no_reply_messages_count)

#     print("Total number of messages: ", total_messages)
#     print("Number messages replied within 5 minutes: ", message_reply_less_than_5_minutes)
#     print("The Fraction of messages replied within 5 minutes: ", message_reply_less_than_5_minutes / total_messages)

#     fig, ax = plt.subplots()
#     colors = cycle(['#FF5733', '#33FF57', '#5733FF', '#FF33E6', '#FFD133'])  # More colorful palette
#     for channel_name, message_timestamps in channels_stamps.items():
#         x_values = [calculate_time_difference_seconds(msg_timestamp, reply_timestamp) for msg_timestamp, reply_timestamp in message_timestamps]
#         y_values = [datetime.utcfromtimestamp(float(msg_timestamp)).hour for msg_timestamp, _ in message_timestamps]

#         current_color = next(colors)
#         ax.scatter(x_values, y_values, label=channel_name, alpha=0.5, color=current_color)

#     ax.set_xlabel('Time Difference (seconds)')
#     ax.set_ylabel('Time of the Day (hour)')
#     ax.set_title('Scatter Plot of Time Difference vs. Time of the Day')

#     # Set background color
#     ax.set_facecolor('#F0F0F0')  # Light gray background
#     fig.set_facecolor('#E0E0E0')  # Slightly darker background

#     # Move the legend outside and add more spacing
#     ax.legend(loc='upper left', bbox_to_anchor=(1, 1), title='Channels', frameon=False)
#     plt.subplots_adjust(right=0.7)  # Adjust the spacing between the plot and the legend

#     plt.show()

# def get_messages_reply_timestamp_from_channel(path_channel):
#     stamps = []
#     no_reply_messages_count = 0

#     # specify path to get json files
#     json_files = [f"{path_channel}/{pos_json}" for pos_json in os.listdir(path_channel) if pos_json.endswith('.json')]
#     combined = []

#     for json_file in json_files:
#         with open(json_file, 'r', encoding="utf8") as slack_data:
#             json_content = json.load(slack_data)
#             combined.extend(json_content)

#     for msg in combined:
#         if 'subtype' not in msg:
#             text = msg.get('text', None)
#             ts = msg.get('ts', None)

#             if 'replies' in msg:
#                 reply_ts_list = [reply.get('ts') for reply in msg['replies']]
#                 if reply_ts_list:
#                     latest_reply_ts = max(reply_ts_list)
#                     stamps.append((ts, latest_reply_ts))
#                 else:
#                     no_reply_messages_count += 1

#     return stamps, no_reply_messages_count

# if __name__ == "__main__":
#     data_loader = SlackDataLoader(path='./../dataset/')
#     channels_data = data_loader.get_channels()
#     process_channels_data(channels_data)

