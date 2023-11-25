import os
import csv
import json
import torch
import argparse
import subprocess
from pathlib import Path

from trainer import Trainer, TrainerArgs

from TTS.config.shared_configs import BaseDatasetConfig
from TTS.tts.datasets import load_tts_samples
from TTS.tts.layers.xtts.trainer.gpt_trainer import GPTArgs, GPTTrainer, GPTTrainerConfig, XttsAudioConfig
from TTS.utils.manage import ModelManager

from core.helpers import (
    merge_voices
)
from moviepy.video.io.VideoFileClip import VideoFileClip
from core.dereverb import MDXNetDereverb
from pydub import AudioSegment

from core.whisperx.asr import load_model, load_audio
from core.whisperx.alignment import load_align_model, align
from core.whisperx.diarize import DiarizationPipeline, assign_word_speakers

with open('config.json', 'r') as f:
    token_config = json.load(f)
    
DEVICE_TYPE = 'cuda' if torch.cuda.is_available() else 'cpu'
LANGS = ['en', 'es', 'fr', 'de', 'it', 'pt', 'pl', 'tr', 'ru', 'nl', 'cs', 'ar', 'zh-cn', 'ja','hu','ko']

def transcribe_audio_extended(audio_file):
    whisper = load_model('large-v2', device=DEVICE_TYPE, compute_type='int8')
    diarize_model = DiarizationPipeline(use_auth_token=token_config['HF_TOKEN'],device=DEVICE_TYPE)
    
    audio = load_audio(audio_file)
    batch_size = 16 
    while 1:
        try:
            result = whisper.transcribe(audio, batch_size=batch_size,chunk_size=15)
        except RuntimeError:
            batch_size //= 2
            if batch_size == 0:
                raise("audio too long to translate,limit in >30mins")
            else:
                print("reset whisper batch_size={}".format(batch_size))
                continue
        break
    language = result['language']
    model_a, metadata = load_align_model(language_code=language, device=DEVICE_TYPE)
    result = align(result['segments'], model_a, metadata, audio, DEVICE_TYPE, return_char_alignments=False)
    print("diarizing ... wait moment")
    diarize_segments = diarize_model(audio)
    result = assign_word_speakers(diarize_segments, result)
    
    whisper, diarize_model,model_a = (None,None,None)
    del whisper, diarize_model,model_a
    torch.cuda.empty_cache()
    return result['segments'], language


def gen_ft_dataset(original_audio_file):
    ft_dataset_path = Path(original_audio_file).parent.joinpath("ft_dataset")
    subprocess.call("rm -rf {}".format(ft_dataset_path), shell=True)
    Path.mkdir(ft_dataset_path,parents=True, exist_ok=True)
    wavs_path = Path(ft_dataset_path).joinpath("wavs")
    Path.mkdir(wavs_path,parents=True, exist_ok=True)
    csv_path = ft_dataset_path.joinpath("metadata.csv")
    
    ## remove noise
    dereverb = MDXNetDereverb(15)
    dereverb_out = dereverb.split(original_audio_file)
    voice_audio = AudioSegment.from_file(dereverb_out['voice_file'], format='wav')
    speakers, lang = transcribe_audio_extended(dereverb_out['voice_file'])
    
    merged_voices = merge_voices(speakers, voice_audio)
    
    num = 0
    for i,speaker in enumerate(speakers):
        if 'id' in speaker:
            voice = merged_voices[speaker['id']]
        else:
            voice = voice_audio[speaker['start'] * 1000: speaker['end'] * 1000]
        
        ## save .wav splited
        voice_wav_name = 'ft_xtts_{}'.format(i)
        voice_wav_path = wavs_path.joinpath('{}.wav'.format(voice_wav_name))
        voice.export(voice_wav_path, format='wav')
        
        text = speaker['text']
        
        ## generate meatadata.csv
        with open(csv_path, "a", newline='', encoding='utf-8') as f:
            writer = csv.writer(f,delimiter='|')
            writer.writerow([voice_wav_name,text,text])
        num += 1
    
    if num > 10:
        train_num = int(num * 0.9)
        val_num = num - train_num
    else:
        train_num = num - 1
        val_num = 1
    csv_shuf_path = csv_path.parent.joinpath('metadata_shuf.csv')
    csv_train_path = csv_path.parent.joinpath('metadata_train.csv')
    csv_val_path = csv_path.parent.joinpath('metadata_val.csv')
    commad = "shuf {} > {} && head -n {} {} > {} && tail -n {} {} > {}\
              ".format(csv_path,csv_shuf_path,train_num,csv_shuf_path
                       ,csv_train_path,val_num,csv_shuf_path,csv_val_path)
    subprocess.call(commad, shell=True)
    
    return os.path.join(ft_dataset_path), lang

def finetune_xtts(speaker_name,speaker_filename,finetune_workpalce,batch_size, is_gen_dataset):
    print("[Step 1] split audio and generate xtts format datasets")
    original_audio_file = Path(speaker_filename).parent.joinpath("audio_from_video.wav")
    if is_gen_dataset == 1:
        if "mp4" in speaker_filename:
            orig_clip = VideoFileClip(speaker_filename)
            orig_clip.audio.write_audiofile(original_audio_file, codec='pcm_s16le')
        else:
            original_audio_file = speaker_filename

        ft_dataset_path,lang = gen_ft_dataset(original_audio_file)

        torch.save((ft_dataset_path,lang), os.path.join(ft_dataset_path,"dataset.pt"))
    else:
        ft_dataset_path = Path(original_audio_file).parent.joinpath("ft_dataset")
        ft_dataset_path,_,lang = torch.load(os.path.join(ft_dataset_path,"dataset.pt"))
    
    
    if "zh" in lang:
        lang = "zh-cn"
    
    if lang not in LANGS:
        raise("language should be in {}".format(str(LANGS)))
    
    
    print("[Step 2] finetune xtts model for enhance custom speaker's voice")
    # Logging parameters
    RUN_NAME = "{}_GPT_XTTS_v2.0_LJSpeech_FT".format(speaker_name)
    PROJECT_NAME = "{}_XTTS_trainer".format(speaker_name)
    DASHBOARD_LOGGER = "tensorboard"
    LOGGER_URI = None

    # Set here the path that the checkpoints will be saved. Default: ./run/training/
    OUT_PATH = os.path.join(finetune_workpalce, "finetuning")

    # Training Parameters
    OPTIMIZER_WD_ONLY_ON_WEIGHTS = True  # for multi-gpu training please make it False
    START_WITH_EVAL = True  # if True it will star with evaluation
    BATCH_SIZE = batch_size  # set here the batch size
    GRAD_ACUMM_STEPS = 252 // BATCH_SIZE  # set here the grad accumulation steps
    # Note: we recommend that BATCH_SIZE * GRAD_ACUMM_STEPS need to be at least 252 for more efficient training. You can increase/decrease BATCH_SIZE but then set GRAD_ACUMM_STEPS accordingly.

    # Define here the dataset that you want to use for the fine-tuning on.
    config_dataset = BaseDatasetConfig(
        formatter="ljspeech",
        dataset_name="ljspeech",
        path=ft_dataset_path,
        meta_file_train = os.path.join(ft_dataset_path,'metadata_train.csv'),
        meta_file_val = os.path.join(ft_dataset_path,'metadata_val.csv'),
        language=lang
    )

    # Add here the configs of the datasets
    DATASETS_CONFIG_LIST = [config_dataset]

    # Define the path where XTTS v2.0.1 files will be downloaded
    CHECKPOINTS_OUT_PATH = os.path.join(OUT_PATH, "XTTS_v2.0_original_model_files/")
    os.makedirs(CHECKPOINTS_OUT_PATH, exist_ok=True)


    # DVAE files
    DVAE_CHECKPOINT_LINK = "https://coqui.gateway.scarf.sh/hf-coqui/XTTS-v2/main/dvae.pth"
    MEL_NORM_LINK = "https://coqui.gateway.scarf.sh/hf-coqui/XTTS-v2/main/mel_stats.pth"

    # Set the path to the downloaded files
    DVAE_CHECKPOINT = os.path.join(CHECKPOINTS_OUT_PATH, os.path.basename(DVAE_CHECKPOINT_LINK))
    MEL_NORM_FILE = os.path.join(CHECKPOINTS_OUT_PATH, os.path.basename(MEL_NORM_LINK))

    # download DVAE files if needed
    if not os.path.isfile(DVAE_CHECKPOINT) or not os.path.isfile(MEL_NORM_FILE):
        print(" > Downloading DVAE files!")
        ModelManager._download_model_files([MEL_NORM_LINK, DVAE_CHECKPOINT_LINK], CHECKPOINTS_OUT_PATH, progress_bar=True)


    # Download XTTS v2.0 checkpoint if needed
    TOKENIZER_FILE_LINK = "https://coqui.gateway.scarf.sh/hf-coqui/XTTS-v2/main/vocab.json"
    XTTS_CHECKPOINT_LINK = "https://coqui.gateway.scarf.sh/hf-coqui/XTTS-v2/main/model.pth"

    # XTTS transfer learning parameters: You we need to provide the paths of XTTS model checkpoint that you want to do the fine tuning.
    TOKENIZER_FILE = os.path.join(CHECKPOINTS_OUT_PATH, os.path.basename(TOKENIZER_FILE_LINK))  # vocab.json file
    XTTS_CHECKPOINT = os.path.join(CHECKPOINTS_OUT_PATH, os.path.basename(XTTS_CHECKPOINT_LINK))  # model.pth file

    # download XTTS v2.0 files if needed
    if not os.path.isfile(TOKENIZER_FILE) or not os.path.isfile(XTTS_CHECKPOINT):
        print(" > Downloading XTTS v2.0 files!")
        ModelManager._download_model_files(
            [TOKENIZER_FILE_LINK, XTTS_CHECKPOINT_LINK], CHECKPOINTS_OUT_PATH, progress_bar=True
        )


    # Training sentences generations
    SPEAKER_REFERENCE = [
        os.path.join(ft_dataset_path,"wavs/ft_xtts_1.wav")  # speaker reference to be used in training test sentences
    ]
    LANGUAGE = config_dataset.language
    
    test_sentences_dict = {
        "en": "It took me quite a long time to develop a voice, and now that I have it I'm not going to be silent.",
        "zh-cn": "开发这个功能掉了好多头发，希望你们喜欢"
    }

    # init args and config
    model_args = GPTArgs(
        max_conditioning_length=132300,  # 6 secs
        min_conditioning_length=66150,  # 3 secs
        debug_loading_failures=False,
        max_wav_length=255995,  # ~11.6 seconds
        max_text_length=200,
        mel_norm_file=MEL_NORM_FILE,
        dvae_checkpoint=DVAE_CHECKPOINT,
        xtts_checkpoint=XTTS_CHECKPOINT,  # checkpoint path of the model that you want to fine-tune
        tokenizer_file=TOKENIZER_FILE,
        gpt_num_audio_tokens=1026,
        gpt_start_audio_token=1024,
        gpt_stop_audio_token=1025,
        gpt_use_masking_gt_prompt_approach=True,
        gpt_use_perceiver_resampler=True,
    )
    # define audio config 22050
    audio_config = XttsAudioConfig(sample_rate=22050, dvae_sample_rate=22050, output_sample_rate=24000)
    # training parameters config
    config = GPTTrainerConfig(
        output_path=OUT_PATH,
        model_args=model_args,
        run_name=RUN_NAME,
        project_name=PROJECT_NAME,
        run_description="""
            GPT XTTS training
            """,
        dashboard_logger=DASHBOARD_LOGGER,
        logger_uri=LOGGER_URI,
        audio=audio_config,
        batch_size=BATCH_SIZE,
        batch_group_size=48,
        eval_batch_size=BATCH_SIZE,
        num_loader_workers=8,
        eval_split_max_size=256,
        print_step=50,
        plot_step=100,
        log_model_step=1000,
        save_step=10000,
        save_n_checkpoints=1,
        save_checkpoints=True,
        # target_loss="loss",
        print_eval=False,
        # Optimizer values like tortoise, pytorch implementation with modifications to not apply WD to non-weight parameters.
        optimizer="AdamW",
        optimizer_wd_only_on_weights=OPTIMIZER_WD_ONLY_ON_WEIGHTS,
        optimizer_params={"betas": [0.9, 0.96], "eps": 1e-8, "weight_decay": 1e-2},
        lr=5e-06,  # learning rate
        lr_scheduler="MultiStepLR",
        # it was adjusted accordly for the new step scheme
        lr_scheduler_params={"milestones": [50000 * 18, 150000 * 18, 300000 * 18], "gamma": 0.5, "last_epoch": -1},
        test_sentences=[
            {
                "text": test_sentences_dict[LANGUAGE],
                "speaker_wav": SPEAKER_REFERENCE,
                "language": LANGUAGE,
            }
        ],
    )

    # init the model from config
    model = GPTTrainer.init_from_config(config)

    # load training samples
    train_samples, eval_samples = load_tts_samples(
        DATASETS_CONFIG_LIST,
        eval_split=True,
        eval_split_max_size=config.eval_split_max_size,
        eval_split_size=config.eval_split_size,
    )

    # init the trainer and 🚀
    
    trainer = Trainer(
         TrainerArgs(
            restore_path=None,  # xtts checkpoint is restored via xtts_checkpoint key so no need of restore it using Trainer restore_path parameter
            skip_train_epoch=False,
            start_with_eval=START_WITH_EVAL,
            grad_accum_steps=GRAD_ACUMM_STEPS,
        ),
        config,
        output_path=OUT_PATH,
        model=model,
        train_samples=train_samples,
        eval_samples=eval_samples,
    )
    trainer.fit()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Finetune xtts model for custom speaker')
    
    parser.add_argument('speaker_name', help="name your custom speaker")
    parser.add_argument('speaker_filename', help="the abslute path to speaker file which contant the speaker's quality voice, can be .mp4 or .wav")
    parser.add_argument('finetune_workpalce', help='the abslute path to save model finetuned, available cache bigger is better')
    parser.add_argument('batch_size', default=3, help='custom the finetuing batch_size')
    parser.add_argument('is_gen_dataset', default=1, help='the abslute path to save model finetuned, available cache bigger is better')
    args = parser.parse_args()
    
    finetune_xtts(
        speaker_name = args.speaker_name,
        speaker_filename=args.speaker_filename,
        finetune_workpalce=args.finetune_workpalce,
        batch_size=int(args.batch_size),
        is_gen_dataset = int(args.is_gen_dataset)
        )