from TTS.tts.configs.xtts_config import XttsConfig
from TTS.tts.models.xtts import Xtts

config = XttsConfig()
config.load_json("D:/Neuro/DeepPavlovTest/DeepPavlovTest/XTTS-v2/config.json")
model = Xtts.init_from_config(config)
model.load_checkpoint(config, checkpoint_dir="D:/Neuro/DeepPavlovTest/DeepPavlovTest/XTTS-v2", eval=True)
model.cuda()


outputs = model.synthesize(
    "Warhammer 40,000 stands as an exceptionally intricate sci-fi fantasy universe.",
    config,
    speaker_wav="D:/Neuro/DeepPavlovTest/DeepPavlovTest/Audio/test.wav",
    gpt_cond_len=3,
    language="en",
)