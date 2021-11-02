from datetime import datetime
from azure.eventgrid import EventGridClient
from msrest.authentication import TopicCredentials
import pytz
import uuid


def publish_event(data):
    credentials = TopicCredentials(
        "Nb3K+Pif1hNKWzk4j7dp6bzbgQywjEiMWGKUs+k8qZk="
    )
    event_grid_client = EventGridClient(credentials)
    event_grid_client.publish_events(
        "nancyeventgrid.eastus-1.eventgrid.azure.net",
        events=[{
            'id': uuid.uuid4(),
            'subject': "FaceData",
            'data': data,
            'event_type': 'NancyEventGrid',
            'event_time': datetime.now(pytz.timezone('US/Central')),
            'data_version': 2
        }]
    )
