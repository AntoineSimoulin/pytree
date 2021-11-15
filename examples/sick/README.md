Implementation of Tai et al., ([2015](#tai-2015))

For the Constituency TreeLSTM, you can run the following script:

```bash
python examples/sick/run_sick_n_ary.py \
    --glove_file_path glove.840B.300d.txt \
    --do_train \
    --do_eval \
    --do_predict \
    --output_dir './model' \
    --dataset_name 'sick' \
    --remove_unused_columns false \
    --learning_rate 0.05 \
    --per_device_train_batch_size 25 \
    --num_train_epochs 10 \
    --weight_decay 1e-4 \
    --lr_scheduler_type constant \
    --overwrite_cache false \
    --overwrite_output_dir \
    --evaluation_strategy steps \
    --load_best_model_at_end true  \
    --eval_steps 177 \
    --save_steps 177 \
    --metric_for_best_model pearson
```

You should get the following results:

```bash
***** train metrics *****
  epoch                    =       15.0
  train_loss               =     0.3237
  train_runtime            = 0:08:00.43
  train_samples            =       4439
  train_samples_per_second =    138.592
  train_steps_per_second   =      5.557

***** eval metrics *****
  epoch                   =       15.0
  eval_loss               =     0.5873
  eval_mse                =    27.7861
  eval_pearson            =    85.3163
  eval_runtime            = 0:00:01.96
  eval_samples            =        495
  eval_samples_per_second =    251.634
  eval_spearman           =     80.371
  eval_steps_per_second   =     31.518

***** predict metrics *****
  predict_samples         =       4906
  test_loss               =     0.5733
  test_mse                =    28.1483
  test_pearson            =    85.3743
  test_runtime            = 0:00:13.78
  test_samples_per_second =    355.835
  test_spearman           =    79.3668
  test_steps_per_second   =     44.534
```

For the Dependency TreeLSTM, you can run the following script:

```bash
python examples/sick/run_sick_child_sum.py \
    --glove_file_path glove.840B.300d.txt \
    --do_train \
    --do_eval \
    --do_predict \
    --output_dir './model' \
    --dataset_name 'sick' \
    --remove_unused_columns false \
    --learning_rate 0.05 \
    --per_device_train_batch_size 25 \
    --num_train_epochs 5 \
    --weight_decay 1e-4 \
    --lr_scheduler_type constant \
    --overwrite_cache true \
    --overwrite_output_dir \
    --evaluation_strategy steps \
    --load_best_model_at_end true  \
    --eval_steps 177 \
    --save_steps 177 \
    --metric_for_best_model pearson
```

You should get the following results:

```bash
***** train metrics *****
  epoch                    =       15.0
  train_loss               =     0.2208
  train_runtime            = 0:05:31.18
  train_samples            =       4439
  train_samples_per_second =    201.051
  train_steps_per_second   =      8.062

***** eval metrics *****
  epoch                   =       15.0
  eval_loss               =     0.5191
  eval_mse                =    24.7928
  eval_pearson            =    87.0975
  eval_runtime            = 0:00:00.85
  eval_samples            =        495
  eval_samples_per_second =    579.243
  eval_spearman           =    82.2074
  eval_steps_per_second   =     72.552

***** predict metrics *****
  predict_samples         =       4906
  test_loss               =      0.513
  test_mse                =    25.7405
  test_pearson            =    86.4682
  test_runtime            = 0:00:06.55
  test_samples_per_second =    748.054
  test_spearman           =    80.4307
  test_steps_per_second   =     93.621
```

## 📚 References

> <div id="tai-2015">Kai Sheng Tai, Richard Socher, Christopher D. Manning <a href=https://aclanthology.org/P15-1150>Improved Semantic Representations From Tree-Structured Long Short-Term Memory Networks.</a> ACL (1) 2015: 1556-1566</div>