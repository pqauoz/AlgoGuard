import pandas as pd

class FeatureExtractor:
    """
    Extracts the 10 raw core features from network flow data.
    """

    RAW_COLUMNS = ['dur', 'spkts', 'dpkts', 'sbytes', 'dbytes', 'rate', 'sttl', 'dttl', 'proto', 'state']

    @staticmethod
    def extract(flow_data):
        """
        Convert incoming JSON dictionary into a Pandas DataFrame with 10 columns.
        """
        raw_features = {
            'dur': float(flow_data.get('dur', 0)),
            'spkts': int(flow_data.get('spkts', 0)),
            'dpkts': int(flow_data.get('dpkts', 0)),
            'sbytes': int(flow_data.get('sbytes', 0)),
            'dbytes': int(flow_data.get('dbytes', 0)),
            'rate': float(flow_data.get('rate', 0)),
            'sttl': int(flow_data.get('sttl', 0)),
            'dttl': int(flow_data.get('dttl', 0)),
            'proto': str(flow_data.get('proto', 'tcp')),  
            'state': str(flow_data.get('state', 'CON'))    
        }
        
        return pd.DataFrame([raw_features], columns=FeatureExtractor.RAW_COLUMNS)