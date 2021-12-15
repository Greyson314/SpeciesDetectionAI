import pickle
import csv
import os
os.chdir("resources/kaggle_datasets/dataset_2/")

with open("history_m31.pkl", "rb") as f:
    history_dict = pickle.load(f)

# print(history_dict.keys())


with open("history_log.csv", "w", newline='', encoding='utf-8') as f:
    writer = csv.writer(f)
    x = 0
    writer.writerow(["Loss", "Accuracy", "False Negative Rate", "False Positive Rate"])
    while x < len(history_dict['loss']):
        # print(x)
        writer.writerow([
            history_dict["loss"][x],
            history_dict["acc"][x],
            history_dict["false_negatives"][x]/896,
            history_dict["false_positives"][x]/896,
        ])
        x+=1



# if make_plot:
#     print("Creating Plot...")
#     fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 5))
#     ax1.plot(epochs, loss_values, "co", label="Training Loss")
#     # ax1.plot(epochs,val_loss_values,'m', label='Validation Loss')
#     ax1.set_title("Training loss")
#     ax1.set_xlabel("Epochs")
#     ax1.set_ylabel("Loss")
#     ax1.legend()

#     ax2.plot(epochs, acc_values, "co", label="Training accuracy")
#     ax2.set_title("Training accuracy")
#     ax2.set_xlabel("Epochs")
#     ax2.set_ylabel("Accuracy")
#     ax2.legend()

#     ax3.plot(epochs, fp_values, "co", label="False Positives")
#     ax3.set_title("False Positives")
#     ax3.set_xlabel("Epochs")
#     ax3.set_ylabel("Frequency")
#     ax3.legend()

#     ax4.plot(epochs, fn_values, "co", label="False Negatives")
#     ax4.set_title("False Negatives")
#     ax4.set_xlabel("Epochs")
#     ax4.set_ylabel("Frequency")
#     ax4.legend()

#     plt.show()
