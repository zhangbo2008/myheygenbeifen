# MyHeyGen | [中文](./README.md)
A civilian version video translation tool that provides a full process solution for voice cloning, translation correction, and lip synthesis
## Thanks
[HeyGenClone](https://github.com/BrasD99/HeyGenClone.git)、[TTS](https://github.com/coqui-ai/tts)、[Video-retalking](https://github.com/OpenTalker/video-retalking)
## Gallery
- finetune效果 [【MyHeyGen测试 | 节选霉霉的NYU毕业演讲片段】]( https://www.bilibili.com/video/BV1vc411X7EA/?share_source=copy_web&vd_source=453c36b4abef37acd389d4c01b149023)
- [【6种语言向世界报喜，我的女儿面面出生啦 ! | MyHeyGen 用例】]( https://www.bilibili.com/video/BV1eC4y1E7qc/?share_source=copy_web&vd_source=453c36b4abef37acd389d4c01b149023)
- [【好家伙一下子学了英语、日语、法语、俄语、韩语5国外语，肾好，肾好！ | MyHeyGen效果演示】](https://www.bilibili.com/video/BV1wC4y1E78h/?share_source=copy_web&vd_source=453c36b4abef37acd389d4c01b149023)
- [【张三老师英文普法！英文区的网友有福啦】](https://www.bilibili.com/video/BV1XN41137Bv/?share_source=copy_web&vd_source=453c36b4abef37acd389d4c01b149023)
- [【MyHeyGen测试|这英的英语倍儿地道！】](https://www.bilibili.com/video/BV1vN4y1D7mo/?share_source=copy_web&vd_source=453c36b4abef37acd389d4c01b149023)
## Tutorial
[【MyHeyGen来了！！！】]( https://www.bilibili.com/video/BV14C4y1J7dY/?share_source=copy_web&vd_source=453c36b4abef37acd389d4c01b149023)

## One click package 
[【MyHeyGen教程|这样配置应该简单很多吧】](https://www.bilibili.com/video/BV1cN4y1D73X/?share_source=copy_web&vd_source=453c36b4abef37acd389d4c01b149023)
Equivalent to a one click package, no environment required, but with small funds

## Environmental preparation
1. Get [huggingface](https://huggingface.co/) Token in config.json `HF_TOKEN`
2. Apply for [fanyi](https://fanyi-api.baidu.com/?fr=pcHeader)APPID and APPKey in config.json `TS_APPID` and `TS_APPKEY`
3. Download [weights](https://drive.google.com/file/d/1dYy24q_67TmVuv_PbChe2t1zpNYJci1J/view?usp=sharing) and unzip it in `MyHeyGen`，download [checkpoints](https://drive.google.com/drive/folders/18rhjMpxK8LVVxf7PI6XwOidt8Vouv_H0?usp=share_link) then put it in MyHeyGen/video-retalking,remenber copy GFPGANv1.4.pth from weights to checkpoints!

<div>
  <figure>
  <img alt='weights path' src="./img/weights.png?raw=true" width="300px"/>
  <img alt='checkpoints path' src="./img/checkpoints.png?raw=true" width="300px"/>
  <figure>
</div>


## Install
```
git clone https://github.com/AIFSH/MyHeyGen.git
cd MyHeyGen
bash install.sh
```
or use docker
```
docker pull registry.cn-beijing.aliyuncs.com/codewithgpu2/aifsh-myheygen:o3U7yjrWg5
```
## Try
```
python translate.py /root/MyHeyGen/test/src.mp4 'zh-cn' -o /root/MyHeyGen/test/out_zh.mp4
```
## Use
```
python translate.py src_video_path lang_code -o out_video_opath
## lang_code in ['en', 'es', 'fr', 'de', 'it', 'pt', 'pl', 'tr', 'ru', 'nl', 'cs', 'ar', 'zh-cn', 'ja','hu','ko']
## Corresponding to [English, Spanish, French, German, Italian, Portuguese, Polish, Turkish, Russian, Dutch, Czech, Arabic, Chinese (Simplified), Japanese, Hungarian, Korean]
```
## Update log
- 2023.11.7  add TTS_MODEL in config.json to custom model
- 2023.11.8 update TTS for more reality
- 2023.11.9 fix video-retalking oface error
- 2023.11.10 fix librosa version conflict with latest TTS

## WeChat Group and Sponsor 
<div>
  <figure>
  <img alt='WeChat Group' src="./img/chat.jpg?raw=true" width="300px"/>
  <img alt='Sponsor' src="./img/ludan.jpg?raw=true" width="300px"/>
  <figure>
</div>


## about `config.json`
```
{
    "DET_TRESH": 0.3, 
    "DIST_TRESH": 0.2,
    "DB_NAME": "storage.db",
    "HF_TOKEN": "",  ## token apply form huggingface
    "TS_APPID": "",  ## Baidu Fanyi 
    "TS_APPKEY": "", ## Baidu Fanyi 
    "HUMAN_TRANS": 0, ## 1 human check; 0 auto
    "SPEAKER_NUM": 1, 
    "TTS_MODEL":"tts_models/multilingual/multi-dataset/xtts_v2",
    "FT_TTS_MODEL": "" ##the finetune model path to enable xtts fineting mode
}
```
## About  Finetune
Test on GPU A5000 24GB
```
python xtts_ft.py luoxiang /root/autodl-tmp/xtts_ft/luoxiang/speaker.WAV /root/autodl-tmp/xtts_ft 3 1

# luoxiang  ---specker id
# /root/autodl-tmp/xtts_ft/luoxiang/speaker.WAV ---.wav,.mp4, > 30mins
# /root/autodl-tmp/xtts_ft ---workplace of xtts finetune to save model checkpoints,>20GB
# 3 ---batch_size
# 1 ---1 generate dataset, 0 no genrate again
```