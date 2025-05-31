from pathlib import Path
from openai import OpenAI
import sounddevice as sd
import librosa
import os
from io import BytesIO

speech_file_path = Path(__file__).parent / "siliconcloud-generated-speech.mp3"

client = OpenAI(
    api_key=os.getenv("SILICONFLOW_API_KEY"), # 从 https://cloud.siliconflow.cn/account/ak 获取
    base_url="https://api.siliconflow.cn/v1"
)

'''
with client.audio.speech.with_streaming_response.create(
  model="FunAudioLLM/CosyVoice2-0.5B", # 支持 fishaudio / GPT-SoVITS / CosyVoice2-0.5B 系列模型
  voice="speech:Yujin:4vqiu837ib:oihcwneshrogxciqngtu", # 用户上传音色名称，参考
  # 用户输入信息
  input="すみません、この電車は渋谷駅に止まりますか？",
  response_format="mp3"
) as response:
    response.stream_to_file(speech_file_path)
'''

with open('text.txt', 'r', encoding='utf-8') as file:
    input_text = file.readlines()
print("Start running.")
for line in input_text:
    response = client.audio.speech.create(
        model="FunAudioLLM/CosyVoice2-0.5B",
        #voice="speech:Jinyu_CHN_Normal:4vqiu837ib:xcmkruokqfbyepjiblgl",
        #voice="speech:Jinyu_CHN_Bright:4vqiu837ib:loehtqaaoztadaygsqle",
        voice="speech:Jinyu_JPN:4vqiu837ib:zskdgcwvglbmkwcsdiqv",
        #voice="speech:Jinyu_ENG:4vqiu837ib:fqjlneqbfmdxydjcbola",
        #voice="speech:Yujin:4vqiu837ib:oihcwneshrogxciqngtu",
        #voice="speech:Yujin_ENGS:4vqiu837ib:zgrcvuegczodmntcavqy",
        #voice="speech:Yujin_ENGL:4vqiu837ib:gnggihujgikbqcohpwoi",
        #voice="speech:Yujin_JPN:4vqiu837ib:ugfziheummmkgwoxwqlm",
        #voice="speech:Yujin_JPN_kawa:4vqiu837ib:aijhlqkautqjltkimvnx",
        input=line,
        response_format="mp3"
    )
    response.write_to_file(speech_file_path)
    print("Start playing.")
    audio_data, sr = librosa.load(BytesIO(response.content),sr=None)
    sd.play(audio_data, sr)
    sd.wait()