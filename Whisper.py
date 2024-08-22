from openai import OpenAI
client = OpenAI()

audio_file = open("speech.mp3", "rb")
transcript = client.audio.transcriptions.create(
  file=audio_file,
  model="whisper-1",
#   response_format="verbose_json",
  response_format="srt",
  timestamp_granularities=["segment"],
  language="ja",
  temperature=0.3
)

print(transcript)
