import os
import logging
import asyncio
import json
import tempfile
from datetime import datetime, timedelta
from typing import Dict, Any, Optional

# ایمپورت‌های لازم برای تلگرام
from telegram import Update
from telegram.ext import Application, MessageHandler, filters, ContextTypes

# ایمپورت‌های لازم برای گوگل
import google.generativeai as genai
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import Flow
from googleapiclient.discovery import build

# سایر کتابخانه‌ها
import speech_recognition as sr
from pydub import AudioSegment
import notion_client
from PIL import Image
import numpy as np
import chromadb

logger = logging.getLogger(__name__)

class VoiceAssistantBot:
    def __init__(self, secrets: Dict[str, str]):
        self.secrets = secrets
        self.notion = notion_client.Client(auth=secrets['notion_key'])
        self.calendar_service = None
        self.recognizer = sr.Recognizer()
        #self.notion_db_properties = {}
        self.collection = None # <-- برای کالکشن کروما

        try:
            genai.configure(api_key=secrets['gemini_api_key'])
            self.gemini_model = genai.GenerativeModel('gemini-1.5-flash')
            self.embedding_model = 'models/embedding-001'
            logging.info("✅ کلاینت Google Gemini با موفقیت راه‌اندازی شد.")
        except Exception as e:
            logging.error(f"❌ خطا در راه‌اندازی کلاینت Gemini: {e}")
            self.gemini_model = None
        # ===== تغییر اصلی: راه‌اندازی ChromaDB Cloud =====
        try:
            logging.info("☁️ در حال اتصال به دیتابیس ابری ChromaDB...")
            chroma_client = chromadb.CloudClient(
                tenant='stonesam669',          # <-- نام Tenant شما از سایت
                database='Second Brain',       # <-- نام دیتابیس شما از سایت
                api_key=secrets['chroma_api_key']
            )
            self.collection = chroma_client.get_or_create_collection("second_brain_collection")
            logging.info(f"✅ با موفقیت به کالکشن '{self.collection.name}' در ChromaDB Cloud متصل شدید.")
        except Exception as e:
            logging.error(f"❌ خطا در اتصال به ChromaDB Cloud: {e}", exc_info=True)
        # ===============================================

    def _discover_notion_db_properties(self, db_id: str):
        if not db_id: return
        try:
            db_info = self.notion.databases.retrieve(database_id=db_id)
            self.notion_db_properties[db_id] = db_info['properties']
            logging.info(f"✅ ساختار دیتابیس {db_id} با موفقیت شناسایی شد.")
        except Exception as e:
            logging.error(f"❌ خطا در شناسایی ساختار دیتابیس {db_id}: {e}")

    def _get_google_auth_creds(self) -> Optional[Credentials]:
        try:
            google_creds_json = json.loads(self.secrets['google_creds'])
            with tempfile.NamedTemporaryFile(mode='w+', delete=False, suffix='.json') as f:
                json.dump(google_creds_json, f)
                creds_path = f.name
            
            flow = Flow.from_client_secrets_file(
                creds_path,
                scopes=['https://www.googleapis.com/auth/calendar'],
                redirect_uri='urn:ietf:wg:oauth:2.0:oob'
            )

            auth_url, _ = flow.authorization_url(prompt='consent')
            print("\n" + "="*70 + "\n🔗 احراز هویت گوگل کلندر\n" + f"1. لطفاً روی لینک زیر کلیک کنید:\n{auth_url}")
            print("\n2. به حساب گوگل خود اجازه دسترسی بدهید و کد را کپی کنید.")
            auth_code = input("3. کد را اینجا وارد کرده و Enter را بزنید: ").strip()
            
            flow.fetch_token(code=auth_code)
            os.unlink(creds_path)
            return flow.credentials
        except Exception as e:
            logging.error(f"❌ خطا در احراز هویت گوگل: {e}", exc_info=True)
            return None

    def setup_google_calendar(self) -> bool:
        print("\n⏳ در حال راه‌اندازی سرویس تقویم گوگل...")
        creds = self._get_google_auth_creds()
        if creds:
            self.calendar_service = build('calendar', 'v3', credentials=creds)
            print("✅ سرویس تقویم گوگل با موفقیت راه‌اندازی شد.")
            return True
        print("❌ راه‌اندازی سرویس تقویم گوگل ناموفق بود.")
        return False

    async def _analyze_text_with_gemini(self, text: str) -> Optional[Dict[str, Any]]:
        if not self.gemini_model:
            logging.error("کلاینت Gemini راه‌اندازی نشده است.")
            return None
        
        logging.info("🤖 در حال پردازش متن با هوش مصنوعی Gemini...")
        prompt = f"""
        # بخش ۱: نقش و هویت (Role-Playing)
        شما یک متخصص جهانی در زمینه مدیریت دانش و تحلیلگر ارشد محتوا با نام "Athena" هستید. شما در تبدیل افکار پراکنده، یادداشت‌های صوتی و متون خام به دانش ساختاریافته، اتمی و قابل اقدام، بهترین در جهان هستید. دقت، ساختارمندی و درک عمیق از نیت کاربر، ویژگی‌های اصلی شماست.

        # بخش ۲: وظیفه اصلی (Clarity and Specificity)
        وظیفه اصلی شما، دریافت یک متن خام (که می‌تواند از OCR یک اسکرین‌شات، تبدیل یک صوت به متن یا یک یادداشت تایپ‌شده باشد) و تبدیل آن به یک فایل JSON کاملاً ساختاریافته بر اساس "اسکیمای دانش جهانی" (UKS) است.
        
        # بخش ۳: تعریف اسکیمای JSON - UKS (Defining the Output Format)
        شما باید اطلاعات استخراج شده را دقیقاً در قالب ساختار JSON زیر قرار دهید:

        ```json
        {
          "core_content": {
            "title": "یک عنوان بسیار کوتاه و توصیفی برای این دانش (حداکثر ۱۰ کلمه).",
            "summary": "یک خلاصه ۱ تا ۳ جمله‌ای که جان کلام متن ورودی را بیان می‌کند.",
            "original_text": "متن کامل و خام ورودی برای آرشیو و بازبینی."
          },
          "source_and_context": {
            "source_type": "نوع منبع، مثلا: Book, Podcast, Article, Video, Conversation, Personal Thought, Screenshot.",
            "source_name": "نام دقیق منبع، مثلا: 'Deep Work', 'Huberman Lab Podcast'. اگر نامشخص بود null قرار بده.",
            "source_author_or_creator": "نام نویسنده یا خالق اثر. اگر نامشخص بود null قرار بده."
          },
          "categorization": {
            "primary_domain": "حوزه اصلی مرتبط با این دانش. فقط یکی از موارد لیست مجاز انتخاب شود.",
            "tags_and_keywords": ["لیستی از برچسب‌ها و کلمات کلیدی دقیق که به جستجوی آینده کمک می‌کند."],
            "entities": ["لیستی از موجودیت‌های خاص نام برده شده مانند اسامی افراد، محصولات، شرکت‌ها و..."]
          },
          "actionability": {
            "actionability_type": "نوع اقدام مرتبط با این دانش. فقط یکی از موارد لیست مجاز انتخاب شود.",
            "action_item_description": "شرح دقیق وظیفه در صورتی که قابل اقدام باشد. در غیر این صورت null قرار بده."
          }
        }
        # بخش ۴: قوانین و محدودیت‌های کلیدی (Constraining the Model)
        برای انجام وظیفه خود، شما موظف به رعایت قوانین زیر هستید:

        خروجی فقط JSON باشد: پاسخ شما باید فقط و فقط یک آبجکت JSON معتبر باشد. هیچ متن، توضیح یا مقدمه‌ای قبل یا بعد از آبجکت JSON ننویسید.

        رعایت لیست‌های مجاز (Enums):

        برای فیلد primary_domain، فقط یکی از این مقادیر را استفاده کن: ["YouTube", "Kaizen (Learning)", "Health & Lifestyle", "Finance (Crypto/Buying)", "Project Management", "Personal Journal (Ikigai)", "Other"]

        برای فیلد actionability_type، فقط یکی از این مقادیر را استفاده کن: ["Actionable Task", "Topic for Research", "Idea for Creation", "Information to Store", "Financial Record", "Personal Reflection"]

        عدم اختراع اطلاعات: هرگز اطلاعاتی که در متن ورودی وجود ندارد را حدس نزن و به خروجی اضافه نکن. اگر اطلاعاتی برای یک فیلد موجود نیست، مقدار آن را null قرار بده.

        زبان خروجی: تمام مقادیر متنی در فایل JSON باید به زبان فارسی باشند، مگر اینکه نام یک موجودیت خاص (مانند "Deep Work") به زبان اصلی باشد.
        # بخش ۵: مثال‌های آموزشی (Few-Shot Learning)
        مثال ۱: ورودی ساده از یک کتاب
        INPUT:
        "یادم باشه از کتاب دیپ ورک کَل نیوپورت این نکته رو برای ویدیوی مدیریت زمانم استفاده کنم که میگه کار عمیق مثل یک ابرقدرته. باید یه وقتی هم بذارم در موردش بیشتر تحقیق کنم."
        JSON OUTPUT:
        {
          "core_content": {
            "title": "کار عمیق به عنوان یک ابرقدرت",
            "summary": "نکته‌ای از کتاب 'کار عمیق' اثر کل نیوپورت که بیان می‌کند توانایی انجام کار عمیق یک مزیت رقابتی و شبیه به یک ابرقدرت در دنیای امروز است. این نکته برای استفاده در ویدیوی مدیریت زمان مناسب است.",
            "original_text": "یادم باشه از کتاب دیپ ورک کَل نیوپورت این نکته رو برای ویدیوی مدیریت زمانم استفاده کنم که میگه کار عمیق مثل یک ابرقدرته. باید یه وقتی هم بذارم در موردش بیشتر تحقیق کنم."
          },
          "source_and_context": {
            "source_type": "Book",
            "source_name": "Deep Work",
            "source_author_or_creator": "Cal Newport"
          },
          "categorization": {
            "primary_domain": "YouTube",
            "tags_and_keywords": ["deep work", "productivity", "time management", "focus"],
            "entities": ["Cal Newport", "Deep Work"]
          },
          "actionability": {
            "actionability_type": "Topic for Research",
            "action_item_description": "تحقیق بیشتر در مورد مفهوم کار عمیق (Deep Work)."
          }
        }
        # مثال ۲: ورودی ترکیبی و پیچیده
        INPUT:
        "مدل جدید نور Aputure Amaran 150c. قیمت: ۵۶۹ دلار. CRI: 95+. قابلیت کنترل با اپلیکیشن. این نور عالیه برای کانال یوتیوبم، چون رنگ پوست رو خیلی طبیعی نشون میده. توی اون بررسی میگفت برای سلامت چشم هم بهتر از نورهای ارزون‌قیمته، چون فلیکر نداره. باید بذارمش توی Buying List و با مدل Elgato Key Light مقایسه کنم. این یه سرمایه‌گذاری برای کارمه."
        JSON OUTPUT:
        {
          "core_content": {
            "title": "بررسی نور Aputure 150c برای یوتیوب و لیست خرید",
            "summary": "نور جدید Aputure Amaran 150c با قیمت ۵۶۹ دلار، یک گزینه عالی برای استودیوی یوتیوب است. این مدل به دلیل شاخص CRI بالای ۹۵، رنگ پوست را طبیعی نمایش می‌دهد و به خاطر تکنولوژی ضد لرزش (Flicker-Free)، برای سلامت چشم نیز مفید است. این مورد باید به عنوان یک سرمایه‌گذاری در لیست خرید ثبت و با رقیبش، Elgato Key Light، مقایسه شود.",
            "original_text": "مدل جدید نور Aputure Amaran 150c. قیمت: ۵۶۹ دلار. CRI: 95+. قابلیت کنترل با اپلیکیشن. این نور عالیه برای کانال یوتیوبم، چون رنگ پوست رو خیلی طبیعی نشون میده. توی اون بررسی میگفت برای سلامت چشم هم بهتر از نورهای ارزون‌قیمته، چون فلیکر نداره. باید بذارمش توی Buying List و با مدل Elgato Key Light مقایسه کنم. این یه سرمایه‌گذاری برای کارمه."
          },
          "source_and_context": {
            "source_type": "Video",
            "source_name": "یک نقد و بررسی آنلاین تجهیزات",
            "source_author_or_creator": null
          },
          "categorization": {
            "primary_domain": "YouTube",
            "tags_and_keywords": ["تجهیزات فیلم‌برداری", "نورپردازی", "سلامت چشم", "لیست خرید", "سرمایه‌گذاری"],
            "entities": ["Aputure Amaran 150c", "Elgato Key Light"]
          },
          "actionability": {
            "actionability_type": "Actionable Task",
            "action_item_description": "مدل نور Aputure 150c را در 'Buying List' ثبت کرده و آن را با مدل 'Elgato Key Light' از نظر قیمت و ویژگی‌ها مقایسه کن."
          }
        }
        # مثال ۳: ورودی شخصی و کوتاه (مورد مرزی)
        INPUT:
        "امروز خیلی احساس خستگی و بی‌انگیزگی می‌کنم. نمی‌دونم چرا."
        JSON OUTPUT:
        {
          "core_content": {
            "title": "احساس خستگی و بی‌انگیزگی امروز",
            "summary": "یادداشتی شخصی در مورد احساس خستگی و بی‌انگیزگی در طی روز بدون دانستن علت مشخص آن.",
            "original_text": "امروز خیلی احساس خستگی و بی‌انگیزگی می‌کنم. نمی‌دونم چرا."
          },
          "source_and_context": {
            "source_type": "Personal Thought",
            "source_name": null,
            "source_author_or_creator": null
          },
          "categorization": {
            "primary_domain": "Personal Journal (Ikigai)",
            "tags_and_keywords": ["احساسات", "خستگی", "بی‌انگیزگی", "جورنالینگ"],
            "entities": []
          },
          "actionability": {
            "actionability_type": "Personal Reflection",
            "action_item_description": null
          }
        }
        # بخش ۶: دستور نهایی و ورودی کاربر
        اکنون، متن خام زیر را پردازش کرده و خروجی JSON مربوطه را تولید کن:

        [<<متن خام ورودی از کاربر اینجا قرار می‌گیرد>>]
        """
        try:
            response = self.gemini_model.generate_content(prompt)
            json_text = response.text.strip().replace("```json", "").replace("```", "")
            return json.loads(json_text)
        except Exception as e:
            logging.error(f"❌ خطا در تحلیل با Gemini: {e}", exc_info=True)
            return None

    async def _create_calendar_event(self, entities: Dict[str, Any]) -> str:
        if not self.calendar_service:
            return "سرویس تقویم در دسترس نیست."
        try:
            summary = entities.get("summary", "رویداد بدون عنوان")
            start_time_str = entities.get("start_time")
            if not start_time_str:
                return "زمان رویداد مشخص نشد."
                
            start_time = datetime.fromisoformat(start_time_str.replace('Z', '+00:00'))
            end_time = start_time + timedelta(hours=1)

            event = {
                'summary': summary,
                'start': {'dateTime': start_time.isoformat(), 'timeZone': 'UTC'},
                'end': {'dateTime': end_time.isoformat(), 'timeZone': 'UTC'},
            }
            created_event = self.calendar_service.events().insert(calendarId='primary', body=event).execute()
            return f"✅ رویداد «{summary}» با موفقیت در تقویم گوگل ثبت شد."
        except Exception as e:
            logging.error(f"❌ خطا در ایجاد رویداد تقویم: {e}", exc_info=True)
            return "مشکلی در ایجاد رویداد تقویم پیش آمد."

    async def _add_to_knowledge_base(self, content: str) -> str:
        db_id = self.secrets.get('notion_kb_db_id')
        if not db_id: return "خطا: شناسه دیتابیس کتابخانه دانش تعریف نشده است."

        try:
            logging.info(f"🧠 در حال ساخت Embedding با مدل {self.embedding_model}...")
            embedding_response = genai.embed_content(
                model=self.embedding_model,
                content=content,
                task_type="RETRIEVAL_DOCUMENT"
            )
            embedding_vector = embedding_response['embedding']
            embedding_str = json.dumps(embedding_vector)
            chunk_size = 2000
            embedding_chunks = [{"text": {"content": chunk}} for chunk in [embedding_str[i:i + chunk_size] for i in range(0, len(embedding_str), chunk_size)]]

            properties = {
                "Name": {"title": [{"text": {"content": content[:150]}}]},
                "Content": {"rich_text": [{"text": {"content": content}}]},
                "Embedding": {"rich_text": embedding_chunks}
            }
            self.notion.pages.create(parent={"database_id": db_id}, properties=properties)
            return "✅ مطلب با موفقیت در کتابخانه دانش نوشن ذخیره شد."
        except Exception as e:
            logging.error(f"❌ خطا در ذخیره در کتابخانه دانش: {e}", exc_info=True)
            return "مشکلی در ذخیره اطلاعات در نوشن پیش آمد."

    async def _query_knowledge_base(self, query: str) -> str:
        db_id = self.secrets.get('notion_kb_db_id')
        if not db_id: return "خطا: شناسه دیتابیس کتابخانه دانش تعریف نشده است."
        
        try:
            logging.info(f"🔎 در حال ساخت Embedding برای پرس‌وجو با {self.embedding_model}...")
            query_embedding_response = genai.embed_content(model=self.embedding_model, content=query, task_type="RETRIEVAL_QUERY")
            query_vector = np.array(query_embedding_response['embedding'])

            all_pages = []
            cursor = None
            while True:
                response = self.notion.databases.query(database_id=db_id, start_cursor=cursor)
                all_pages.extend(response.get('results', []))
                if not response.get('has_more'): break
                cursor = response.get('next_cursor')
            
            if not all_pages: return "پایگاه دانش شما خالی است."
            logging.info(f"✅ {len(all_pages)} صفحه از کتابخانه دانش بازیابی شد.")

            page_similarities = []
            for page in all_pages:
                props = page.get('properties', {})
                embedding_chunks = props.get("Embedding", {}).get('rich_text', [])
                if not embedding_chunks: continue
                embedding_json = "".join([chunk.get('text', {}).get('content', '') for chunk in embedding_chunks])
                try:
                    doc_vector = np.array(json.loads(embedding_json))
                    similarity = np.dot(query_vector, doc_vector) / (np.linalg.norm(query_vector) * np.linalg.norm(doc_vector))
                    page_content = props.get("Content", {}).get('rich_text', [{}])[0].get('text', {}).get('content', 'محتوا موجود نیست')
                    page_similarities.append((similarity, page_content))
                except (json.JSONDecodeError, ValueError) as e:
                    logging.warning(f"⚠️ خطا در پردازش Embedding برای صفحه {page.get('id')}: {e}")
                    continue
            
            page_similarities.sort(key=lambda x: x[0], reverse=True)
            relevant_docs = [doc[1] for doc in page_similarities if doc[0] > 0.7][:3]
            
            if not relevant_docs: return "متاسفانه مطلب مرتبطی در پایگاه دانش شما پیدا نکردم."
            logging.info(f"✅ {len(relevant_docs)} نکته مرتبط برای پاسخ‌گویی یافت شد.")

            context_str = "\n\n---\n\n".join(relevant_docs)
            final_prompt = f"شما یک دستیار هوش مصنوعی هستید که بر اساس پایگاه دانش شخصی کاربر به سوالات پاسخ می‌دهید. بر اساس «متن‌های مرتبط» زیر، به «سوال کاربر» یک پاسخ جامع و دقیق به زبان فارسی بدهید.\n\n**متن‌های مرتبط از پایگاه دانش:**\n{context_str}\n\n**سوال کاربر:**\n{query}\n\n**پاسخ شما (به فارسی):**"
            
            logging.info("✍️ در حال تولید پاسخ نهایی با Gemini...")
            final_response = self.gemini_model.generate_content(final_prompt)
            return final_response.text
        except Exception as e:
            logging.error(f"❌ خطا در فرآیند پرس‌وجو: {e}", exc_info=True)
            return f"یک خطای غیرمنتظره در هنگام جستجو رخ داد: {e}"

    async def _process_user_request(self, text: str, update: Update):
        await update.message.reply_chat_action('typing')
        analysis = await self._analyze_text_with_gemini(text)
        
        if not analysis:
            await update.message.reply_text("❌ مشکلی در ارتباط با سرویس هوش مصنوعی (Gemini) پیش آمد.")
            return

        intent = analysis.get("intent")
        entities = analysis.get("entities", {})

        if intent == "CALENDAR_EVENT":
            response_text = await self._create_calendar_event(entities)
            await update.message.reply_text(response_text)
        elif intent == "KNOWLEDGE_STORAGE":
            content = entities.get("content", text)
            response_text = await self._add_to_knowledge_base(content)
            await update.message.reply_text(response_text)
        elif intent == "QUERY":
            query = entities.get("query", text)
            await update.message.reply_text("🔎 در حال جستجو در پایگاه دانش شما...")
            answer = await self._query_knowledge_base(query)
            await update.message.reply_text(answer)
        else:
            await update.message.reply_text("🤔 متوجه منظور شما نشدم. لطفاً واضح‌تر بیان کنید.")

    async def _convert_voice_to_text(self, voice_file_path: str) -> str:
        logging.info("🎵 در حال تبدیل صدا به متن...")
        try:
            audio = AudioSegment.from_ogg(voice_file_path)
            wav_path = voice_file_path + ".wav"
            audio.export(wav_path, format="wav")
            with sr.AudioFile(wav_path) as source:
                audio_data = self.recognizer.record(source)
            text = self.recognizer.recognize_google(audio_data, language='fa-IR')
            os.remove(wav_path)
            return text
        except Exception as e:
            logging.error(f"❌ خطا در تبدیل صدا به متن: {e}", exc_info=True)
            if 'wav_path' in locals() and os.path.exists(wav_path):
                os.remove(wav_path)
            return ""

    async def handle_voice_message(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        await update.message.reply_text("🎤 پیام صوتی دریافت شد. لطفاً صبر کنید...")
        voice = update.message.voice
        with tempfile.NamedTemporaryFile(suffix=".ogg", delete=False) as temp_file:
            voice_file = await context.bot.get_file(voice.file_id)
            await voice_file.download_to_drive(temp_file.name)
            voice_path = temp_file.name
        
        text = await self._convert_voice_to_text(voice_path)
        os.unlink(voice_path)

        if text:
            await update.message.reply_text(f"📝 متن شناسایی شده: «{text}»")
            await self._process_user_request(text, update)
        else:
            await update.message.reply_text("❌ متاسفانه نتوانستم صدایتان را تشخیص دهم.")

    async def handle_text_message(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        user_text = update.message.text
        logging.info(f"⌨️ پیام متنی دریافت شد: '{user_text}'")
        await self._process_user_request(user_text, update)

    async def handle_photo_message(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        await update.message.reply_text("🖼️ تصویر دریافت شد. در حال استخراج متن با Gemini...")
        photo_file = await context.bot.get_file(update.message.photo[-1].file_id)
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as temp_photo:
            await photo_file.download_to_drive(temp_photo.name)
            photo_path = temp_photo.name
        
        try:
            img = Image.open(photo_path)
            response = self.gemini_model.generate_content(["Extract all text from this image in Persian.", img])
            extracted_text = response.text
            os.unlink(photo_path)
            
            if extracted_text and extracted_text.strip():
                await update.message.reply_text(f"📝 متن استخراج شده:\n\n«{extracted_text}»\n\n📚 در حال افزودن به کتابخانه دانش...")
                response_text = await self._add_to_knowledge_base(extracted_text)
                await update.message.reply_text(response_text)
            else:
                await update.message.reply_text("متنی در تصویر یافت نشد.")
        except Exception as e:
            logging.error(f"❌ خطا در پردازش تصویر: {e}", exc_info=True)
            await update.message.reply_text("مشکلی در پردازش تصویر پیش آمد.")
            if os.path.exists(photo_path):
                os.unlink(photo_path)

    async def run(self):
        logging.info("\n🚀 در حال راه‌اندازی ربات تلگرام...")
        app = Application.builder().token(self.secrets['telegram']).build()
        
        app.add_handler(MessageHandler(filters.VOICE, self.handle_voice_message))
        app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, self.handle_text_message))
        app.add_handler(MessageHandler(filters.PHOTO, self.handle_photo_message))
        
        logging.info("🔥 ربات با موفقیت فعال شد! آماده دریافت پیام.")
        await app.run_polling()
