# Test Enhancer

```
TE=True python -m swebench.test_enhancer.testgen --run_id <id> --instance_id <instance_id>
TE=True python -m swebench.test_enhancer.testgen --run_id TE_1 --instance_id django__django-12915
```

```
python -m swebench.harness.run_evaluation --dataset_name princeton-nlp/SWE-bench --predictions_path gold --max_workers 4 --run_id TE_1 --instance_ids django__django-12915
python -m swebench.harness.run_evaluation --dataset_name princeton-nlp/SWE-bench --predictions_path <path_to_preds.jsonl> --max_workers 4 --run_id TE_1 --instance_ids django__django-12915
```
