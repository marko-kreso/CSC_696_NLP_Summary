from datasets import load_dataset

dataset = load_dataset("billsum", split="ca_test")

billsum = billsum.train_test_split(test_size=.2)

billsum["train"][0]

