from datasets import load_dataset

billsum = load_dataset("billsum", split="ca_test")

billsum = billsum.train_test_split(test_size=.2)

print(billsum["train"][0])

