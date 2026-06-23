from training.train import train_one_run
from evaluation.metrics import compute_metrics
from sklearn.metrics import accuracy_score

import torch
import numpy as np
import config as config

def run_experiment():
    seeds = [1, 7, 21, 42, 123]

    results =[]

    for seed in seeds:
        model , test_loader = train_one_run(seed)

        model.eval()
        y_true=[]
        y_pred=[]
        y_probs=[]
        

        with torch.no_grad():
            for X , y in test_loader:
                X= X.to(config.DEVICE)

                output = model(X)
                probs = torch.softmax(output,dim=1)
                preds = output.argmax(dim=1)

                y_true.append(y.cpu())
                y_pred.append(preds.cpu())
                y_probs.append(probs.cpu())

        y_true = torch.cat(y_true)
        y_pred = torch.cat(y_pred)
        y_probs = torch.cat(y_probs)


        cm, report , auc = compute_metrics(y_true.numpy() , y_pred.numpy()  , y_probs.numpy() )   
            
        acc = accuracy_score(y_true.numpy(), y_pred.numpy())

        results.append(acc)

        print(f"\n--- Seed {seed} Results ---")
        print(f"Accuracy: {acc:.4f}")
        print(f"ROC-AUC : {auc:.4f}")

        print("\nConfusion Matrix:")
        print(cm)

        print("\nClassification Report:")
        print(report)

    print("\nMean Accuracy:", np.mean(results))
    print("Std:", np.std(results))  