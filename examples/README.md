Implementation of ([Tai et al., 2015](#tai-2015))


```bash
python pytree/examples/run_sick.py \
    --glove_file_path ./glove.6B.300d.txt \
    --do_train \
    --do_eval \
    --output_dir './model' \
    --dataset_name 'sick' \
    --remove_unused_columns False \
    --learning_rate 0.025 \
    --per_device_train_batch_size 25 \
    --num_train_epochs 20
```

```
CUDA_VISIBLE_DEVICES=2 python examples/run_sick.py     --glove_file_path /data/asimouli/GLOVE/glove.6B.300d.txt     --do_train     --do_eval      --output_dir './model'     --dataset_name 'sick'     --remove_unused_columns False     --learning_rate 0.05     --per_device_train_batch_size 25     --num_train_epochs 15    --weight_decay 1e-4  --lr_scheduler_type constant  --do_predict    --overwrite_cache True  --overwrite_output_dir
```

## References

> <div id="tai-2015">Kai Sheng Tai, Richard Socher, Christopher D. Manning <a href=https://aclanthology.org/P15-1150>Improved Semantic Representations From Tree-Structured Long Short-Term Memory Networks.</a> ACL (1) 2015: 1556-1566</div>