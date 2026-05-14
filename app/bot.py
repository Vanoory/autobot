from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

from telegram import (
    InlineKeyboardButton,
    InlineKeyboardMarkup,
    KeyboardButton,
    ReplyKeyboardMarkup,
    Update,
)
from telegram.constants import ChatAction, ParseMode
from telegram.ext import (
    Application,
    CallbackQueryHandler,
    CommandHandler,
    ContextTypes,
    MessageHandler,
    filters,
)

from app.config import Config, load_config
from app.database import Database, Draft
from app.services.content import ContentService
from app.services.knowledge import KnowledgeBase
from app.services.llm import MultiLLM
from app.services.market import MarketService
from app.services.rendering import Renderer
from app.services.telegram_format import format_post_html, plain_text_for_caption


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
LOGGER = logging.getLogger(__name__)


MENU = ReplyKeyboardMarkup(
    [
        [KeyboardButton("Тест: доброе утро"), KeyboardButton("Тест: новости")],
        [KeyboardButton("Тест: сигнал"), KeyboardButton("Черновики")],
        [KeyboardButton("Статус"), KeyboardButton("Помощь")],
    ],
    resize_keyboard=True,
)


@dataclass(slots=True)
class Services:
    config: Config
    database: Database
    content: ContentService
    market: MarketService
    knowledge: KnowledgeBase


def draft_keyboard(draft_id: int) -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup(
        [
            [
                InlineKeyboardButton("Одобрить", callback_data=f"draft:approve:{draft_id}"),
                InlineKeyboardButton("Редактировать", callback_data=f"draft:edit:{draft_id}"),
            ],
            [InlineKeyboardButton("Отклонить", callback_data=f"draft:reject:{draft_id}")],
        ]
    )


def owner_only(func):
    async def wrapper(update: Update, context: ContextTypes.DEFAULT_TYPE, *args, **kwargs):
        services: Services = context.application.bot_data["services"]
        user = update.effective_user
        if user is None or user.id != services.config.owner_id:
            if update.effective_message:
                await update.effective_message.reply_text("Этот бот настроен только для владельца канала.")
            return
        return await func(update, context, *args, **kwargs)

    return wrapper


def _render_draft_preview(draft: Draft, header: str | None = None) -> str:
    formatted_post = format_post_html(draft.kind, draft.text, draft.payload)
    return (
        f"<b>{header or 'Новый черновик'}</b>\n"
        f"<code>{draft.kind} #{draft.id}</code>\n\n"
        f"{formatted_post}"
    )


async def send_draft_preview(
    application: Application,
    services: Services,
    draft: Draft,
    *,
    header: str | None = None,
) -> None:
    preview_text = _render_draft_preview(draft, header)
    if draft.image_path and Path(draft.image_path).exists():
        with Path(draft.image_path).open("rb") as photo:
            await application.bot.send_photo(
                chat_id=services.config.owner_id,
                photo=photo,
                caption=plain_text_for_caption(preview_text),
                parse_mode=ParseMode.HTML,
                reply_markup=draft_keyboard(draft.id),
            )
    else:
        await application.bot.send_message(
            chat_id=services.config.owner_id,
            text=preview_text,
            parse_mode=ParseMode.HTML,
            reply_markup=draft_keyboard(draft.id),
        )


async def create_and_send_draft(
    application: Application,
    services: Services,
    *,
    kind: str,
    text: str,
    image_path: str | None,
    payload: dict[str, Any],
    header: str | None = None,
) -> Draft:
    draft = await services.database.create_draft(kind=kind, text=text, image_path=image_path, payload=payload)
    await send_draft_preview(application, services, draft, header=header)
    return draft


@owner_only
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.effective_message.reply_text(
        "Бот запущен. Все автопосты сначала будут приходить тебе на одобрение.",
        reply_markup=MENU,
    )


@owner_only
async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.effective_message.reply_text(
        "Что умею:\n"
        "- делать утренние посты\n"
        "- собирать новостную карточку\n"
        "- искать сигнал и рисовать график\n"
        "- принимать txt/md/pdf/docx для базы знаний\n"
        "- отправлять все сначала тебе на проверку\n\n"
        "Если нажмешь 'Редактировать', следующим сообщением можно прислать новый текст для черновика.",
        reply_markup=MENU,
    )


@owner_only
async def status_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    services: Services = context.application.bot_data["services"]
    pending = await services.database.list_pending_drafts()
    open_signals = await services.database.signal_count("open")
    docs = await services.database.knowledge_documents()
    await update.effective_message.reply_text(
        "Статус бота:\n"
        f"- черновиков в ожидании: {len(pending)}\n"
        f"- открытых сигналов на мониторинге: {open_signals}\n"
        f"- документов в базе знаний: {len(docs)}",
        reply_markup=MENU,
    )


@owner_only
async def list_drafts(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    services: Services = context.application.bot_data["services"]
    drafts = await services.database.list_pending_drafts()
    if not drafts:
        await update.effective_message.reply_text("Сейчас нет черновиков в ожидании.", reply_markup=MENU)
        return
    for draft in drafts:
        await send_draft_preview(context.application, services, draft, header="Черновик из очереди")


@owner_only
async def text_router(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    message = update.effective_message
    if message is None or not message.text:
        return

    services: Services = context.application.bot_data["services"]
    pending_edit_id = context.user_data.get("pending_edit_draft_id")
    if pending_edit_id:
        draft = await services.database.get_draft(pending_edit_id)
        if draft is None:
            context.user_data.pop("pending_edit_draft_id", None)
            await message.reply_text("Черновик для редактирования уже не найден.", reply_markup=MENU)
            return
        await services.database.update_draft(draft.id, text=message.text)
        context.user_data.pop("pending_edit_draft_id", None)
        updated = await services.database.get_draft(draft.id)
        if updated:
            await send_draft_preview(context.application, services, updated, header="Черновик обновлен")
        await message.reply_text("Новый текст сохранен.", reply_markup=MENU)
        return

    text = message.text.strip()
    if text == "Тест: доброе утро":
        await generate_morning_draft(context.application, services)
        await message.reply_text("Утренний пост собрал и отправил тебе выше.", reply_markup=MENU)
        return
    if text == "Тест: новости":
        await generate_news_draft(context.application, services)
        await message.reply_text("Новостную карточку подготовил и отправил тебе выше.", reply_markup=MENU)
        return
    if text == "Тест: сигнал":
        created = await generate_signal_draft(context.application, services)
        response = (
            "Сетап подготовил и отправил тебе выше."
            if created
            else "Сейчас не нашел достаточно чистого сетапа."
        )
        await message.reply_text(response, reply_markup=MENU)
        return
    if text == "Черновики":
        await list_drafts(update, context)
        return
    if text == "Статус":
        await status_command(update, context)
        return
    if text == "Помощь":
        await help_command(update, context)
        return

    await message.reply_text(
        "Если это не редактирование черновика, лучше пользуйся кнопками ниже.",
        reply_markup=MENU,
    )


@owner_only
async def handle_document(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    message = update.effective_message
    if message is None or message.document is None:
        return

    services: Services = context.application.bot_data["services"]
    await context.bot.send_chat_action(chat_id=message.chat_id, action=ChatAction.UPLOAD_DOCUMENT)
    telegram_file = await message.document.get_file()
    content = await telegram_file.download_as_bytearray()
    try:
        path, text_len = await services.knowledge.store_document(message.document.file_name, bytes(content))
    except Exception as exc:  # noqa: BLE001
        await message.reply_text(f"Не смог обработать файл: {exc}", reply_markup=MENU)
        return

    await message.reply_text(
        f"Файл добавлен в базу знаний.\nИмя: {path.name}\nСимволов текста: {text_len}",
        reply_markup=MENU,
    )


@owner_only
async def callback_router(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    query = update.callback_query
    if query is None or not query.data:
        return
    await query.answer()

    services: Services = context.application.bot_data["services"]
    _, action, draft_id_text = query.data.split(":", 2)
    draft_id = int(draft_id_text)
    draft = await services.database.get_draft(draft_id)
    if draft is None:
        if query.message and query.message.photo:
            await query.edit_message_caption(caption="Черновик уже не найден.")
        else:
            await query.edit_message_text("Черновик уже не найден.")
        return

    if action == "approve":
        await publish_draft(context.application, services, draft)
        await _replace_query_text(query, f"Черновик #{draft.id} опубликован в канал.")
        return
    if action == "reject":
        await services.database.update_draft(draft.id, status="rejected")
        await _replace_query_text(query, f"Черновик #{draft.id} отклонен.")
        return
    if action == "edit":
        context.user_data["pending_edit_draft_id"] = draft.id
        await query.message.reply_text(
            f"Пришли следующим сообщением новый текст для черновика #{draft.id}.",
            reply_markup=MENU,
        )


async def _replace_query_text(query, text: str) -> None:
    if query.message is None:
        return
    if query.message.photo:
        await query.edit_message_caption(caption=text)
    else:
        await query.edit_message_text(text=text)


async def publish_draft(application: Application, services: Services, draft: Draft) -> None:
    formatted_post = format_post_html(draft.kind, draft.text, draft.payload)
    message_id: int | None = None
    if draft.image_path and Path(draft.image_path).exists():
        with Path(draft.image_path).open("rb") as photo:
            message = await application.bot.send_photo(
                chat_id=services.config.channel_id,
                photo=photo,
                caption=plain_text_for_caption(formatted_post),
                parse_mode=ParseMode.HTML,
            )
            message_id = message.message_id
    else:
        message = await application.bot.send_message(
            chat_id=services.config.channel_id,
            text=formatted_post,
            parse_mode=ParseMode.HTML,
        )
        message_id = message.message_id

    await services.database.update_draft(draft.id, status="published")
    await services.database.add_published_post(
        kind=draft.kind,
        text=draft.text,
        image_path=draft.image_path,
        channel_message_id=message_id,
        payload=draft.payload,
    )
    if draft.kind == "signal":
        await services.database.create_tracked_signal(draft.payload)


async def generate_morning_draft(application: Application, services: Services) -> None:
    text, payload = await services.content.generate_morning_post()
    await create_and_send_draft(
        application,
        services,
        kind="morning",
        text=text,
        image_path=None,
        payload=payload,
    )


async def generate_news_draft(application: Application, services: Services) -> None:
    text, image_path, payload = await services.content.generate_news_post()
    await create_and_send_draft(
        application,
        services,
        kind="news",
        text=text,
        image_path=image_path,
        payload=payload,
    )


async def generate_signal_draft(application: Application, services: Services) -> bool:
    open_signals = await services.database.signal_count("open")
    if open_signals >= services.config.max_open_signals:
        return False
    blocked_symbols = await services.database.signal_symbols("open")
    blocked_symbols.update(await services.database.pending_signal_symbols())
    available_symbols = [symbol for symbol in services.config.signal_symbols if symbol.upper() not in blocked_symbols]
    if not available_symbols:
        return False
    signal = await services.content.generate_signal_post(available_symbols)
    if signal is None:
        return False
    text, image_path, payload = signal
    await create_and_send_draft(
        application,
        services,
        kind="signal",
        text=text,
        image_path=image_path,
        payload=payload,
    )
    return True


async def morning_job(context: ContextTypes.DEFAULT_TYPE) -> None:
    services: Services = context.application.bot_data["services"]
    await generate_morning_draft(context.application, services)


async def news_job(context: ContextTypes.DEFAULT_TYPE) -> None:
    services: Services = context.application.bot_data["services"]
    await generate_news_draft(context.application, services)


async def signal_scan_job(context: ContextTypes.DEFAULT_TYPE) -> None:
    services: Services = context.application.bot_data["services"]
    await generate_signal_draft(context.application, services)


async def price_monitor_job(context: ContextTypes.DEFAULT_TYPE) -> None:
    services: Services = context.application.bot_data["services"]
    signals = await services.database.open_signals()
    for signal in signals:
        try:
            price = await services.market.fetch_price(signal.symbol)
        except Exception:
            continue
        if services.market.stop_hit(signal.payload, price):
            await services.database.close_signal(signal.id, "stopped")
            continue
        if signal.target1_announced:
            continue
        if services.market.take1_hit(signal.payload, price):
            text, payload = await services.content.generate_take1_update(signal.payload)
            await create_and_send_draft(
                context.application,
                services,
                kind="take1",
                text=text,
                image_path=None,
                payload=payload,
                header="Сработал первый тейк",
            )
            await services.database.mark_signal_target1_announced(signal.id)


def parse_hhmm(value: str) -> tuple[int, int]:
    hour, minute = value.split(":")
    return int(hour), int(minute)


def schedule_jobs(application: Application, services: Services) -> None:
    tz = ZoneInfo(services.config.timezone)
    morning_h, morning_m = parse_hhmm(services.config.morning_post_time)
    news_h, news_m = parse_hhmm(services.config.news_post_time)

    application.job_queue.run_daily(
        morning_job,
        time=datetime.now(tz).replace(hour=morning_h, minute=morning_m, second=0, microsecond=0).timetz(),
        name="morning_post",
    )
    application.job_queue.run_daily(
        news_job,
        time=datetime.now(tz).replace(hour=news_h, minute=news_m, second=0, microsecond=0).timetz(),
        name="news_post",
    )
    application.job_queue.run_repeating(
        signal_scan_job,
        interval=services.config.signal_scan_interval_minutes * 60,
        first=30,
        name="signal_scan",
    )
    application.job_queue.run_repeating(
        price_monitor_job,
        interval=services.config.price_monitor_interval_minutes * 60,
        first=90,
        name="price_monitor",
    )


async def post_init(application: Application) -> None:
    services: Services = application.bot_data["services"]
    await services.database.init()
    schedule_jobs(application, services)
    LOGGER.info("Bot initialized and jobs scheduled.")


def build_application() -> Application:
    config = load_config()
    database = Database(config.db_path)
    market = MarketService(config.market_data_providers)
    knowledge = KnowledgeBase(database, config.storage_dir)
    renderer = Renderer(config.storage_dir)
    llm = MultiLLM(config)
    content = ContentService(config, database, llm, knowledge, market, renderer)
    services = Services(config=config, database=database, content=content, market=market, knowledge=knowledge)

    application = Application.builder().token(config.bot_token).post_init(post_init).build()
    application.bot_data["services"] = services
    application.add_handler(CommandHandler("start", start))
    application.add_handler(CommandHandler("menu", start))
    application.add_handler(CommandHandler("help", help_command))
    application.add_handler(MessageHandler(filters.Document.ALL, handle_document))
    application.add_handler(CallbackQueryHandler(callback_router, pattern=r"^draft:"))
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, text_router))
    return application


def run() -> None:
    try:
        asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    application = build_application()
    LOGGER.info("Starting bot polling...")
    application.run_polling(drop_pending_updates=True)
