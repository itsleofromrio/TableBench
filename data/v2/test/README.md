---
language:
- en
license: apache-2.0
pretty_name: TableBench
size_categories:
- n<1K
task_categories:
- question-answering
task_ids: []
tags:
- table-question-answering
configs:
  - config_name: table_bench
    data_files: 
    - split: test
      path: "*.jsonl"
---

# Dataset Card for TableBench

## Dataset Summary

TableBench is a dataset that covers 4 major categories and 18 subcategories, focusing on the multi-dimensional capabilities of table question answering.

## Data Fields

| ID | String | Description |
|----|--------|-------------|
| id | string | Unique Identifier |
| qtype | string | Question Type (FactChecking, NumericalReasoning, DataAnalysis, Visualization) |
| qsubtype | string | Question Subtype |
| instruction | string | Instruction to prompt LLM |
| instruction_type | string | Three different instruction types in TableBench: TCoT(Textual Chain of Thought),SCoT(Symbolic Chain of Thought) and PoT(Program of Thought) |
| table | string | Table |
| question | string | Question |
| answer | string | Answer |
| answer_formatter | string | Constraints on Answer Output Format |


## Data Example
An example of 'validation' looks as follows:
```
{
    "id": "60670a8d9b1e39dd845fb1639d0d8b86",
    "qtype": "DataAnalysis",
    "qsubtype": "StatisticalAnalysis",
    "instruction": "You are a data analyst proficient in Python ...",
    "instruction_type": "PoT",
    "table": "{"columns": ["rank", "circuit", "headquarters", "screens", "sites"], "data": [[1, "regal entertainment group", "knoxville , tn", 7367, 580], [2, "amc entertainment inc", "kansas city , mo", 5894, 483], [3, "cinemark theatres", "plano , tx", 3895, 298], [4, "carmike cinemas , inc", "columbus , ga", 2242, 232], [5, "cineplex entertainment", "toronto , on", 1438, 133], [6, "rave motion pictures", "dallas , tx", 939, 62], [7, "marcus theatres", "milwaukee , wi", 687, 55], [8, "national amusements", "dedham , ma", 450, 34], [9, "empire theatres", "stellarton , ns", 438, 53]]}",
    "question": "Can you calculate the standard deviation of the number of screens operated by the top 5 movie theater chains?",
    "answer": "2472.33",
    "answer_formatter": "The generated Python code should follow the format below, and ensure the first two code lines is exactly the same with the following code block:\n[Python Code Format]\n```python\nimport pandas as pd \ndf = pd.read_csv('table.csv')\n...\nprint(f'Final Answer: {{answer}}')\n```\n\nEnsure the final answer is the last line in python code and can only be in the \"print(f'Final Answer: {{answer}}')\" form, no other from. Ensure variable \"answer\" can only be \"AnswerName1, AnswerName2...\" form, no other form, and \"AnswerName\" can only be a number or entity name, as short as possible, without any explanation."
}
```

## Data Usage
- If you wish to directly assess the capabilities of LLMs on tabular data, you can utilize `TableBench-PoT`, `TableBench-SCoT`, and `TableBench-TCoT` to evaluate the model's abilities directly. 
- If you prefer to customize the prompt method for evaluation, please adhere to the specifications in the `answer_formatter` to reduce evaluation errors caused by inconsistent free-form answers.

## Citation
If you use the data from this project, please cite the original paper:
```
@article{wu2024tablebench,
  title={TableBench: A Comprehensive and Complex Benchmark for Table Question Answering},
  author={Wu, Xianjie and Yang, Jian and Chai, Linzheng and Zhang, Ge and Liu, Jiaheng and Du, Xinrun and Liang, Di and Shu, Daixin and Cheng, Xianfu and Sun, Tianzhen and others},
  journal={arXiv preprint arXiv:2408.09174},
  year={2024}
}
```