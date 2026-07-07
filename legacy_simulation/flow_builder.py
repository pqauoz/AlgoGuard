class FlowBuilder:
    """
    Simulates building a network flow from raw packet data for the demo.
    """
    @staticmethod
    def build_from_json(json_data):
        """
        Ensures the incoming JSON has all required fields with default fallbacks.
        """
        return {
            'dur': json_data.get('dur', 0.001),
            'spkts': json_data.get('spkts', 1),
            'dpkts': json_data.get('dpkts', 1),
            'sbytes': json_data.get('sbytes', 100),
            'dbytes': json_data.get('dbytes', 100),
            'rate': json_data.get('rate', 1.0),
            'sttl': json_data.get('sttl', 64),
            'dttl': json_data.get('dttl', 64),
            'proto': json_data.get('proto', 'tcp'),
            'state': json_data.get('state', 'CON')
        }