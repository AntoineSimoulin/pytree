Implementation of ([Tai et al., 2015](#tai-2015))

For the Constituency TreeLSTM, you can run the following script:

```bash
python examples/run_sick_n_ary.py \
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
    --evaluation_strategy epoch
```

You should get the following results:

```bash
***** predict metrics *****
  predict_samples         =       4906
  test_loss               =     0.6236
  test_mse                =    31.8074
  test_pearson            =    83.2404
  test_runtime            = 0:00:13.02
  test_samples_per_second =    376.716
  test_spearman           =    77.1604
  test_steps_per_second   =     47.147
```

For the Dependency TreeLSTM, you can run the following script:

```bash
python examples/run_sick_child_sum.py \
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
    --overwrite_output_dir
```

You should get the following results:

```bash
***** predict metrics *****
  predict_samples         =       4906
  test_loss               =     0.5228
  test_mse                =    26.4252
  test_pearson            =    86.3953
  test_runtime            = 0:00:05.74
  test_samples_per_second =    854.158
  test_spearman           =    80.3738
  test_steps_per_second   =      106.9
```

## References

> <div id="tai-2015">Kai Sheng Tai, Richard Socher, Christopher D. Manning <a href=https://aclanthology.org/P15-1150>Improved Semantic Representations From Tree-Structured Long Short-Term Memory Networks.</a> ACL (1) 2015: 1556-1566</div>