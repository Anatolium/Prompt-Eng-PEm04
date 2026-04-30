import asyncio
import json
import os
import base64
import aiohttp

from aiogram import Bot, Dispatcher
from aiogram.types import Message
from aiogram.filters import Command
from aiogram.fsm.context import FSMContext
from aiogram.fsm.storage.memory import MemoryStorage

from openai import OpenAI
from dotenv import load_dotenv
from ai_prompts import *

load_dotenv()

API_KEY = os.getenv("PROXYAPI_KEY")
BASE_URL = os.getenv("OPENAI_BASE_URL")
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")

# ====== CONFIG ======
client = OpenAI(api_key=API_KEY, base_url=BASE_URL)

if not TELEGRAM_TOKEN:
    raise ValueError("TELEGRAM_TOKEN не найден в .env")

bot = Bot(token=TELEGRAM_TOKEN)

dp = Dispatcher(storage=MemoryStorage())

MAX_PHOTOS = 3

# ====== UTILS ======
def encode_image(image_bytes):
    return base64.b64encode(image_bytes).decode("utf-8")


# 🔥 Скачивание БЕЗ прокси
async def download_file_direct(file_path: str):
    url = f"https://api.telegram.org/file/bot{TELEGRAM_TOKEN}/{file_path}"

    async with aiohttp.ClientSession() as session:
        async with session.get(url, timeout=30) as resp:
            if resp.status != 200:
                raise Exception(f"Ошибка скачивания файла: {resp.status}")
            return await resp.read()


# ====== HANDLERS ======
@dp.message(Command("start"))
async def cmd_start(message: Message):
    await message.answer(
        "🏠 Привет!\n"
        "1. Отправь голосовое описание квартиры\n"
        "2. Затем до 3 фото\n\n"
        "Я создам объявление для ЦИАН"
    )


# --- AUDIO (универсальный) ---
@dp.message(lambda message: message.voice or message.audio or message.document)
async def handle_any_audio(message: Message, state: FSMContext):
    try:
        file_id = None

        if message.voice:
            file_id = message.voice.file_id
        elif message.audio:
            file_id = message.audio.file_id
        elif message.document:
            if message.document.mime_type and message.document.mime_type.startswith("audio"):
                file_id = message.document.file_id
            else:
                return

        if not file_id:
            await message.answer("⚠️ Не удалось определить аудио.")
            return

        file = await bot.get_file(file_id)

        # 🔥 скачиваем БЕЗ прокси
        file_bytes = await download_file_direct(file.file_path)

        temp_path = "temp_audio.mp3"

        with open(temp_path, "wb") as f:
            f.write(file_bytes)

        with open(temp_path, "rb") as audio_file:
            transcription = client.audio.transcriptions.create(
                model="whisper-1",
                file=audio_file
            )

        text = transcription.text

        await state.update_data(voice_text=text, photos=[])

        await message.answer(
            "🎤 Описание получено!\n"
            "Теперь отправь до 3 фото квартиры."
        )

    except Exception as e:
        import traceback
        traceback.print_exc()
        await message.answer(f"⚠️ Ошибка при обработке аудио: {str(e)}")


# --- PHOTO ---
@dp.message(lambda message: message.photo)
async def handle_photo(message: Message, state: FSMContext):
    data = await state.get_data()
    voice_text = data.get("voice_text")

    if not voice_text:
        await message.answer("⚠️ Сначала отправь голосовое описание.")
        return

    photos = data.get("photos", [])

    if len(photos) >= MAX_PHOTOS:
        await message.answer("⚠️ Максимум 3 фото.")
        return

    photo = message.photo[-1]
    file = await bot.get_file(photo.file_id)

    # 🔥 скачиваем БЕЗ прокси
    file_bytes = await download_file_direct(file.file_path)
    image_data = file_bytes

    photos.append(image_data)
    await state.update_data(photos=photos)

    if len(photos) < MAX_PHOTOS:
        await message.answer(f"📸 Фото {len(photos)}/{MAX_PHOTOS}. Добавь ещё.")
        return

    await message.answer("🔄 Обрабатываю...")

    try:
        vision = await analyze_with_vision(photos)
        structured = await extract_structured_data(voice_text, vision)
        listing = await generate_listing(structured)

        await message.answer(listing, parse_mode="HTML")

        await state.clear()

    except Exception:
        await message.answer("⚠️ Ошибка обработки.")


# ====== AI FUNCTIONS ======
async def analyze_with_vision(photos):
    content = [
        {"type": "text", "text": VISION_PROMPT},
        *[
            {
                "type": "input_image",
                "image_base64": encode_image(photo)
            }
            for photo in photos
        ]
    ]

    response = client.responses.create(
        model="gpt-4.1",
        input=[ # type: ignore
            {"role": "user", "content": content}
        ]
    )

    return response.output_text


async def extract_structured_data(voice_text, vision_analysis):
    response = client.responses.create( # type: ignore
        model="gpt-4.1",
        response_format={"type": "json_object"},
        input=[
            {"role": "system", "content": STRUCTURE_PROMPT},
            {
                "role": "user",
                "content": f"Голос:\n{voice_text}\n\nФото:\n{vision_analysis}"
            }
        ]
    )

    return json.loads(response.output_text)


async def generate_listing(structured_data):
    response = client.responses.create(
        model="gpt-4.1",
        input=[ # type: ignore
            {"role": "system", "content": GENERATION_PROMPT},
            {
                "role": "user",
                "content": json.dumps(structured_data, ensure_ascii=False)
            }
        ],
        temperature=0.7
    )

    return response.output_text


# ====== START ======
async def main():
    delay = 5

    while True:
        try:
            print("🚀 Бот запускается...")
            await dp.start_polling(bot)
            delay = 5

        except Exception as e:
            print(f"🌐 Ошибка сети: {e}")
            print(f"⏳ Повтор через {delay} сек...")
            await asyncio.sleep(delay)
            delay = min(delay * 2, 60)


if __name__ == "__main__":
    asyncio.run(main())
