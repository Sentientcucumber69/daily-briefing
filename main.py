import os
import io
import ssl
import smtplib
import datetime
from email.message import EmailMessage

import feedparser
from gtts import gTTS
from openai import OpenAI

# ---------- CONFIG ----------

# How many articles per feed to consider
ARTICLES_PER_FEED = 8

# Approximate target word count for 10–15 minutes of audio
TARGET_WORD_COUNT = 2000

# RSS feeds (you can tweak these later)
RSS_FEEDS = {
    "Economy & Markets": [
        "https://feeds.reuters.com/reuters/businessNews",
        "https://feeds.reuters.com/reuters/worldNews",
    ],
    "World & Geopolitics": [
        "https://feeds.reuters.com/Reuters/worldNews",
        "https://www.abc.net.au/news/feed/51120/rss.xml",
    ],
    "Technology": [
        "https://feeds.feedburner.com/TechCrunch/",
    ],
    "Gaming": [
        "https://feeds.ign.com/ign/all",
        "https://www.eurogamer.net/?format=rss",
    ],
    "Australia": [
        "https://www.abc.net.au/news/feed/45910/rss.xml",
    ],
}

# Email display name and base subject (date will be appended)
EMAIL_FROM_NAME = "Daily News Briefing"
EMAIL_SUBJECT_BASE = "Daily News Briefing"

# ---------- HELPERS ----------

def get_today_au_date_str() -> str:
    """Return today's date in DD/MM/YYYY for Sydney time (AEDT/AEST)."""
    # GitHub Actions runs in UTC; we approximate Sydney as UTC+10/11.
    # For simplicity, use UTC+11 (AEDT) – good enough for school term.
    now_utc = datetime.datetime.utcnow()
    now_sydney = now_utc + datetime.timedelta(hours=11)
    return now_sydney.strftime("%d/%m/%Y")


def fetch_articles():
    """Fetch recent articles from RSS feeds and return a structured list."""
    sections = []

    for section_name, feeds in RSS_FEEDS.items():
        section_articles = []
        for feed_url in feeds:
            parsed = feedparser.parse(feed_url)
            for entry in parsed.entries[:ARTICLES_PER_FEED]:
                title = getattr(entry, "title", "").strip()
                summary = getattr(entry, "summary", "").strip()
                link = getattr(entry, "link", "").strip()
                if not title:
                    continue
                section_articles.append(
                    {
                        "title": title,
                        "summary": summary,
                        "link": link,
                    }
                )
        if section_articles:
            sections.append(
                {
                    "section": section_name,
                    "articles": section_articles,
                }
            )

    return sections


def build_llm_prompt(sections):
    """Build a prompt for GPT‑4o mini to create a long-form audio briefing."""
    today_str = get_today_au_date_str()

    # Compact representation of articles
    lines = [f"Date: {today_str}", "", "Here are news items grouped by section:"]
    for sec in sections:
        lines.append(f"\nSECTION: {sec['section']}")
        for art in sec["articles"]:
            lines.append(f"- Title: {art['title']}")
            if art["summary"]:
                lines.append(f"  Summary: {art['summary'][:400]}")
            if art["link"]:
                lines.append(f"  Link: {art['link']}")

    articles_text = "\n".join(lines)

    system_msg = (
        "You are an expert news editor and radio host. "
        "Your job is to create a clear, engaging, long-form spoken news briefing "
        "for a high school student on their morning commute in Sydney, Australia. "
        "Assume they are smart but busy: they want context, not fluff."
    )

    user_msg = f"""
Using the news items below, create a 10–15 minute spoken-style news briefing.

Requirements:
- Tone: calm, clear, neutral, like a public radio news bulletin.
- Audience: a senior high school student in Sydney on the light rail to school.
- Structure:
  1. Very short intro (1–2 sentences) with the date and a quick overview.
  2. Economy & Markets (if relevant).
  3. World & Geopolitics.
  4. Technology.
  5. Gaming & Entertainment (if relevant).
  6. Australia-specific news.
  7. A short 'What to watch today' closing section.

- Style:
  - Speak in the first person plural ("we'll look at...", "let's turn to...").
  - Use smooth transitions between sections.
  - Explain key terms briefly when they first appear.
  - Avoid bullet points; write as continuous spoken paragraphs.
  - Do NOT mention URLs or say 'according to this article' – just present the news.

- Length:
  - Aim for around {TARGET_WORD_COUNT} words.
  - It's okay to be a bit shorter or longer if it sounds natural.

Now here is the news data:

{articles_text}
"""

    return system_msg, user_msg


def generate_briefing_text(sections: list) -> str:
    """Call GPT‑4o mini to generate the long-form briefing text."""
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY not set")

    client = OpenAI(api_key=api_key)

    system_msg, user_msg = build_llm_prompt(sections)

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_msg},
        ],
        temperature=0.7,
        max_tokens=4096,
    )

    return response.choices[0].message.content.strip()


def text_to_speech_mp3(text: str) -> bytes:
    """Convert text to MP3 bytes using gTTS, splitting if needed."""
    # gTTS can handle fairly long text, but we’ll be safe and chunk it.
    max_chars = 4000
    chunks = []
    current = []

    for paragraph in text.split("\n"):
        if len("\n".join(current) + "\n" + paragraph) > max_chars and current:
            chunks.append("\n".join(current))
            current = [paragraph]
        else:
            current.append(paragraph)
    if current:
        chunks.append("\n".join(current))

    # Generate MP3 in memory and concatenate
    combined = io.BytesIO()
    for i, chunk in enumerate(chunks):
        tts = gTTS(text=chunk, lang="en")
        buf = io.BytesIO()
        tts.write_to_fp(buf)
        buf.seek(0)
        combined.write(buf.read())

    return combined.getvalue()


def send_email_with_attachment(mp3_bytes: bytes, briefing_text: str):
    """Send the MP3 as an email attachment via Gmail SMTP."""
    gmail_address = os.environ.get("GMAIL_ADDRESS")
    gmail_app_password = os.environ.get("GMAIL_APP_PASSWORD")

    if not gmail_address or not gmail_app_password:
        raise RuntimeError("GMAIL_ADDRESS or GMAIL_APP_PASSWORD not set")

    today_str = get_today_au_date_str()
    subject = f"{EMAIL_SUBJECT_BASE} - {today_str}"

    msg = EmailMessage()
    msg["From"] = f"{EMAIL_FROM_NAME} <{gmail_address}>"
    msg["To"] = gmail_address
    msg["Subject"] = subject

    msg.set_content(
        f"Here is your daily AI-generated news briefing for {today_str}.\n\n"
        "You can listen to the attached MP3 on your commute.\n\n"
        "Text version (for reference):\n\n"
        + briefing_text[:4000]  # avoid huge emails
    )

    filename = f"daily_news_briefing_{today_str.replace('/', '-')}.mp3"
    msg.add_attachment(
        mp3_bytes,
        maintype="audio",
        subtype="mpeg",
        filename=filename,
    )

    context = ssl.create_default_context()
    with smtplib.SMTP_SSL("smtp.gmail.com", 465, context=context) as server:
        server.login(gmail_address, gmail_app_password)
        server.send_message(msg)


def main():
    print("Fetching articles...")
    sections = fetch_articles()
    if not sections:
        raise RuntimeError("No articles fetched from RSS feeds")

    print("Generating briefing text with GPT‑4o mini...")
    briefing_text = generate_briefing_text(sections)

    print("Converting text to speech...")
    mp3_bytes = text_to_speech_mp3(briefing_text)

    print("Sending email...")
    send_email_with_attachment(mp3_bytes, briefing_text)

    print("Done.")


if __name__ == "__main__":
    main()
