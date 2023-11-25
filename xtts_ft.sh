python xtts_ft.py luoxiang /root/autodl-tmp/xtts_ft/luoxiang/speaker.WAV /root/autodl-tmp/xtts_ft 3 1

# luoxiang 说话人编号
# /root/autodl-tmp/xtts_ft/luoxiang/speaker.WAV 语料路径，支持.wav,.mp4文件，建议时长30min以上，音质佳，杂音少
# /root/autodl-tmp/xtts_ft 这是fine-tune工作路径，建议可用存储空间在20GB以上
# 3 这是fine-tune的batch_size
# 1 这里指定是否生成fine-tune所需的dataset，填 0 则不需要再次生成