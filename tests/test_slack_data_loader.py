import unittest
import pandas as pd
from src.loader import SlackDataLoader

class TestSlackParser(unittest.TestCase):
    def setUp(self):
        self.slack_data_loader = SlackDataLoader('dataset') 

    def test_slack_parser(self):

        path_channel = 'dataset/all-week2'
        df = self.slack_data_loader.slack_parser(path_channel)

        self.assertFalse(df.empty)
        expected_columns = ['msg_type', 'msg_content', 'sender_name', 'msg_sent_time', 'msg_dist_type',
                            'time_thread_start', 'reply_count', 'reply_users_count', 'reply_users', 'tm_thread_end',
                            'channel']
        self.assertListEqual(list(df.columns), expected_columns)

    def test_parse_slack_reaction(self):
        
        path = 'dataset/all-week2'
        channel = 'all-week2'

        df_reaction = self.slack_data_loader.parse_slack_reaction(path, channel)
        self.assertFalse(df_reaction.empty)

        expected_columns = ['reaction_name', 'reaction_count', 'reaction_users_count', 'message', 'user_id', 'channel']
        self.assertListEqual(list(df_reaction.columns), expected_columns)

if __name__ == '__main__':
    unittest.main()