from TTS.api import TTS
from core.temp_manager import TempFileManager
import torch
import torchaudio
from TTS.tts.configs.xtts_config import XttsConfig
from TTS.tts.models.xtts import Xtts


class VoiceCloner:
    def __init__(self, config,lang_code):
        # self.api = TTS(f'tts_models/{lang_code}/fairseq/vits')
        self.config = config
        self.lang_code = lang_code
        ft_tts_model = config['FT_TTS_MODEL']
        if len(ft_tts_model) > 1:
            # Add here the xtts_config path
            CONFIG_PATH = "{}/config.json".format(ft_tts_model)
            # Add here the vocab file that you have used to train the model
            TOKENIZER_PATH = "{}/vocab.json".format(ft_tts_model)
            # Add here the checkpoint that you want to do inference with
            XTTS_CHECKPOINT = "{}/best_model.pth".format(ft_tts_model)
            
            print("Loading model {}".format(ft_tts_model))
            config = XttsConfig()
            config.load_json(CONFIG_PATH)
            self.model = Xtts.init_from_config(config)
            self.model.load_checkpoint(config, checkpoint_path=XTTS_CHECKPOINT, vocab_path=TOKENIZER_PATH, use_deepspeed=False)
            if torch.cuda.is_available():
                self.model.cuda()
            
        else:
            self.api = TTS(config["TTS_MODEL"],gpu=torch.cuda.is_available())
            print("TTS model {} Loaded".format(config["TTS_MODEL"]))
    
    def process(self, speaker_wav_filename, text, out_filename=None):
        temp_manager = TempFileManager()
        if not out_filename:
            out_filename = temp_manager.create_temp_file(suffix='.wav').name
            
        if len(self.config['FT_TTS_MODEL']) > 1:
            print("Computing speaker latents...")
            gpt_cond_latent, speaker_embedding = self.model.get_conditioning_latents(
                audio_path=speaker_wav_filename)

            print("Inference...")
            out = self.model.inference(
                text,
                self.lang_code,
                gpt_cond_latent,
                speaker_embedding,
                temperature=0.7, # Add custom parameters here
            )
            torchaudio.save(out_filename, 
                            torch.tensor(out["wav"]).unsqueeze(0), 
                            24000,bits_per_sample=16)
        else:
            self.api.tts_to_file(
                text,
                speaker_wav=speaker_wav_filename,
                file_path=out_filename,
                language=self.lang_code
            )
        return out_filename