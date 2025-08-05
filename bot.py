import os
import logging
import asyncio
import json
import tempfile
from datetime import datetime, timedelta
from typing import Dict, Any, Optional

import openai
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import Flow
from googleapiclient.discovery import build
import speech_recognition as sr
from pydub import AudioSegment
import notion_client
from PIL import Image
import numpy as np

logger = logging.getLogger(__name__)

class VoiceAssistantBot:
    def __init__(self, secrets: Dict[str, str]):
        # ... (محتوای کامل کلاس از پاسخ قبلی که کد کامل را داشتید، اینجا قرار می‌گیرد) ...
        # این شامل تمام متدهای _discover_notion_db_properties, _get_google_auth_creds, 
        # setup_google_calendar, _create_calendar_event, و غیره است.
