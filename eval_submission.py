
import pandas as pd


VERSION = "v1"
TEST_LANG = "English"
UNSEEN_TEST_LANG = "English" if TEST_LANG == "Urdu" else "Urdu"


csv_gt = pd.read_csv(f"{VERSION}_{TEST_LANG}_test_key_dict.csv")
csv_submission = pd.read_csv(f"submission_{VERSION}_{TEST_LANG}_{TEST_LANG}.csv")
csv_merge = csv_submission.merge(csv_gt[["key", "label"]], on="key")

accuracy = (csv_merge["p3"] == csv_merge["label"]).mean()
print("P3 Acc: ", accuracy)

accuracy = (csv_merge["p4"] == csv_merge["label"]).mean()
print("P4 Acc: ", accuracy)

csv_gt = pd.read_csv(f"{VERSION}_{UNSEEN_TEST_LANG}_test_key_dict.csv")
csv_submission = pd.read_csv(f"submission_{VERSION}_{TEST_LANG}_{UNSEEN_TEST_LANG}.csv")
csv_merge = csv_submission.merge(csv_gt[["key", "label"]], on="key")

accuracy = (csv_merge["p5"] == csv_merge["label"]).mean()
print("P5 Acc: ", accuracy)

accuracy = (csv_merge["p6"] == csv_merge["label"]).mean()
print("P6 Acc: ", accuracy)


