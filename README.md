# MNAR In Recommender System

This git is a small experiment by Peng Cao, used for research project in unimelb of Master of Data Science.

We mainly referenced the code from Yuta Saito's paper "Asymmetric Tri-training for Debiasing Missing-Not-At-Random Explicit Feedback":
```python=
@inproceedings{saito2020asymmetric,
  title={Asymmetric tri-training for debiasing missing-not-at-random explicit feedback},
  author={Saito, Yuta},
  booktitle={Proceedings of the 43rd International ACM SIGIR Conference on Research and Development in Information Retrieval},
  year={2020}
}
```

The experiment is generating a different distribution dataset from original Coat dataset, to explore the tri-training framework. The tritraining folder is exactly the same as the one from Yuta's repo, and we use the code carefully with files inside the modified folder. 

We provided here for convenient but please download coat dataset and Yahoo! R3 dataset with permission in:

```python=
Coat: https://www.cs.cornell.edu/~schnabts/mnar/
Yahoo! R3: https://webscope.sandbox.yahoo.com/
```