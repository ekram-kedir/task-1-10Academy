import datetime
import glob
import json
import os
import sys
from collections import Counter
from collections import defaultdict

import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from nltk.corpus import stopwords


def break_combined_weeks(combined_weeks):
    """
    Breaks combined weeks into separate weeks.
    
    Args:
        combined_weeks: list of tuples of weeks to combine
        
    Returns:
        tuple of lists of weeks to be treated as plus one and minus one
    """
    plus_one_week = []
    minus_one_week = []

    for week in combined_weeks:
        if week[0] < week[1]:
            plus_one_week.append(week[0])
            minus_one_week.append(week[1])
        else:
            minus_one_week.append(week[0])
            plus_one_week.append(week[1])

    return plus_one_week, minus_one_week

def get_msgs_df_info(df):
    msgs_count_dict = df.user.value_counts().to_dict()
    replies_count_dict = dict(Counter([u for r in df.replies if r != None for u in r]))
    mentions_count_dict = dict(Counter([u for m in df.mentions if m != None for u in m]))
    links_count_dict = df.groupby("user").link_count.sum().to_dict()
    return msgs_count_dict, replies_count_dict, mentions_count_dict, links_count_dict



def get_messages_dict(msgs):
    msg_list = {
            "msg_id":[],
            "text":[],
            "attachments":[],
            "user":[],
            "mentions":[],
            "emojis":[],
            "reactions":[],
            "replies":[],
            "replies_to":[],
            "ts":[],
            "links":[],
            "link_count":[]
            }


    for msg in msgs:
        if "subtype" not in msg:
            try:
                msg_list["msg_id"].append(msg["client_msg_id"])
            except:
                msg_list["msg_id"].append(None)
            
            msg_list["text"].append(msg["text"])
            msg_list["user"].append(msg["user"])
            msg_list["ts"].append(msg["ts"])
            
            if "reactions" in msg:
                msg_list["reactions"].append(msg["reactions"])
            else:
                msg_list["reactions"].append(None)

            if "parent_user_id" in msg:
                msg_list["replies_to"].append(msg["ts"])
            else:
                msg_list["replies_to"].append(None)

            if "thread_ts" in msg and "reply_users" in msg:
                msg_list["replies"].append(msg["replies"])
            else:
                msg_list["replies"].append(None)
            
            if "blocks" in msg:
                emoji_list = []
                mention_list = []
                link_count = 0
                links = []
                
                for blk in msg["blocks"]:
                    if "elements" in blk:
                        for elm in blk["elements"]:
                            if "elements" in elm:
                                for elm_ in elm["elements"]:
                                    
                                    if "type" in elm_:
                                        if elm_["type"] == "emoji":
                                            emoji_list.append(elm_["name"])

                                        if elm_["type"] == "user":
                                            mention_list.append(elm_["user_id"])
                                        
                                        if elm_["type"] == "link":
                                            link_count += 1
                                            links.append(elm_["url"])


                msg_list["emojis"].append(emoji_list)
                msg_list["mentions"].append(mention_list)
                msg_list["links"].append(links)
                msg_list["link_count"].append(link_count)
            else:
                msg_list["emojis"].append(None)
                msg_list["mentions"].append(None)
                msg_list["links"].append(None)
                msg_list["link_count"].append(0)
    
    return msg_list

def from_msg_get_replies(msg):

    replies = []
    if "thread_ts" in msg and "replies" in msg:
        try:
            for reply in msg["replies"]:
                reply["thread_ts"] = msg["thread_ts"]
                reply["message_id"] = msg["client_msg_id"]
                replies.append(reply)
        except:
            pass
    return replies

def from_msg_get_replies_text(msg):
    '''
    Extract replies from a message
    '''
    replies = []
    if "text" in msg:
        try:
            for reply in msg["replies"]:
                replies.append(msg['text'])
        except:
            pass
        
    return replies

def from_msg_get_replies_text_with_specific_date(msg):
    '''
    Extract replies from a message
    '''
    replies = defaultdict(list)
    if "text" in msg:
        try:
            for reply in msg["replies"]:
                replies[msg['ts']].append(msg['text'])
        except:
            pass
        
    return replies
    
def msgs_to_df(msgs):

    msg_list = get_messages_dict(msgs)
    df = pd.DataFrame(msg_list)
    return df

def process_msgs(msg):
    '''
    select important columns from the message
    '''

    keys = ["client_msg_id", "type", "text", "user", "ts", "team", 
            "thread_ts", "reply_count", "reply_users_count"]
    msg_list = {k:msg[k] for k in keys}
    rply_list = from_msg_get_replies(msg)

    return msg_list, rply_list

def get_messages_from_channel(channel_path):
    '''
    Get all the messages from a channel        
    '''
    channel_json_files = os.listdir(channel_path)
    channel_msgs = [json.load(open(channel_path + "/" + f)) for f in channel_json_files]

    df = pd.concat([pd.DataFrame(msgs) for msgs in channel_msgs])
    
    return df


def convert_2_timestamp(column, data):
    """convert from unix time to readable timestamp
        args: column: columns that needs to be converted to timestamp
                data: data that has the specified column
    """
    if column in data.columns.values:
        timestamp_ = []
        for time_unix in data[column]:
            if time_unix == 0:
                timestamp_.append(0)
            else:
                a = datetime.datetime.fromtimestamp(float(time_unix))
                timestamp_.append(a.strftime('%Y-%m-%d %H:%M:%S'))
        return timestamp_
    else: print(f"{column} not in data")

def get_messages_reply_timestamp_from_channel(channel_path):
    
    json_files = [
        f"{channel_path}/{pos_json}" 
        for pos_json in os.listdir(channel_path) 
        if pos_json.endswith('.json')
    ]     
    combined = []

    for json_file in json_files:
        with open(json_file, 'r', encoding="utf8") as slack_data:
            json_content = json.load(slack_data)
            combined.extend(json_content)
    
    message_time_stamps = []
    no_reply_messages_count = 0

    for msgs in combined:
        if "latest_reply" in msgs:
            message_time_stamps.append([msgs["ts"], msgs["latest_reply"]])
        else:
            no_reply_messages_count += 1

    return message_time_stamps, no_reply_messages_count

def get_channel_messages_replies_timestamp(channel_path):
    json_files = [
        f"{channel_path}/{pos_json}" 
        for pos_json in os.listdir(channel_path) 
        if pos_json.endswith('.json')
    ]   

    combined = []

    for json_file in json_files:
        with open(json_file, 'r', encoding="utf8") as slack_data:
            json_content = json.load(slack_data)
            combined.extend(json_content)
    
    reply_timestamps = []

    for msg in combined:    
        msg_replies = from_msg_get_replies(msg) 
        if msg_replies:
            reply_timestamps.extend([reply['ts'] for reply in msg_replies])

    return reply_timestamps

def get_all_events_timestamp_on_channel(channel_path):
    json_files = [
        f"{channel_path}/{pos_json}" 
        for pos_json in os.listdir(channel_path) 
        if pos_json.endswith('.json')
    ]
    combined = []

    for json_file in json_files:
        with open(json_file, 'r', encoding="utf8") as slack_data:
            json_content = json.load(slack_data)
            combined.extend(json_content)
    
    channel_events_time_stamp = get_timestamps_from_messages(combined)
                 
    return channel_events_time_stamp

def get_timestamps_from_messages(messages):
    timestamps = []

    for msg in messages:
        if 'ts' in msg:
            timestamps.append(msg['ts'])

    return timestamps

def find_reaction_timestamps(channel_path):
    json_files = [
        f"{channel_path}/{pos_json}" 
        for pos_json in os.listdir(channel_path) 
        if pos_json.endswith('.json')
    ]  

    combined = []

    for json_file in json_files:
        with open(json_file, 'r', encoding="utf8") as slack_data:
            json_content = json.load(slack_data)
            combined.extend(json_content)
            
    reaction_timestamps = []
    for item in combined:
        if 'reactions' in item:
            reactions = item['reactions']
            for reaction in reactions:
                if 'ts' in reaction:
                    reaction_timestamps.append(float(reaction['ts']))

    return reaction_timestamps

def get_messages_timestamp_from_channel(channel_path):
    
    json_files = [
        f"{channel_path}/{pos_json}" 
        for pos_json in os.listdir(channel_path) 
        if pos_json.endswith('.json')
    ]     
    combined = []

    for json_file in json_files:
        with open(json_file, 'r', encoding="utf8") as slack_data:
            json_content = json.load(slack_data)
            combined.extend(json_content)
    
    message_time_stamps = defaultdict(list)

    for msgs in combined:
        message_time_stamps[msgs['ts']].append(msgs['text'])

    return message_time_stamps