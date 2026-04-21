import os, time, json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from copy import deepcopy



EPOCHS_PER_STEP = 50
AFTER_TRAIN_EPOCHS = 20
LEARNING_RATE = 0.05
TUBE_OFFSET = 0.1
ALIGN_STRENGTH = 0.4
ANGLE_LIMIT_DEG = 45
RESULTS_ROOT = "data/dynamic/train/updated"


def safe_mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def load_graphs(path):
    with open(path, "r") as f:
        return json.load(f)

def predict_gk(window, slopes, w):
    """TODO: 14 weights: 12 linear + 2 non-linear tweaks"""
    slopes = np.array(slopes[-12:])
    linear = np.dot(slopes, w[:12])
    nl1 = w[12] * np.mean(slopes**2)
    nl2 = w[13] * slopes[-1]*slopes[-2] if len(slopes)>=2 else 0
    return window[-1] + linear + nl1 + nl2

def l2_l1_pull(pred_points, true_start, true_end):
    x = np.arange(len(pred_points))
    m = (true_end - true_start)/(len(pred_points)-1) if len(pred_points)>1 else 0
    l1 = true_start + m*x
    return pred_points + ALIGN_STRENGTH*(l1 - pred_points), l1

def tube_projection(points, center_line):
    upper = center_line + TUBE_OFFSET
    lower = center_line - TUBE_OFFSET
    projected = points.copy()
    projected[points>upper] = upper[points>upper]
    projected[points<lower] = lower[points<lower]
    return projected, upper, lower
# newly added angle strain component for better prediction in lower dimensions
def angle_strain(points):
    out = [points[0]]
    for i in range(1,len(points)):
        slope_prev = out[i-1] - (out[i-2] if i>1 else out[i-1])
        slope_now = points[i] - out[i-1]
        if slope_prev == 0:
            out.append(points[i])
            continue
        angle = np.degrees(np.arctan(abs((slope_now - slope_prev)/(1 + slope_now*slope_prev + 1e-8))))
        if angle > ANGLE_LIMIT_DEG:
            adj = out[i-1] + 0.5*(points[i]-out[i-1])
            out.append(adj)
        else:
            out.append(points[i])
    return np.array(out)


def train_single_graph(graph):
    x_all = np.array(graph["x"], dtype=float)
    y_all = np.array(graph["y"], dtype=float)
    slopes_all = np.array(graph["slopes"], dtype=float)
    n = len(y_all)
    #assert n >= 13, "Each graph must have at least 13 points"

    preds = list(y_all[:12])
    weight_sets = []
    w = np.random.randn(14)*0.01

    for t in range(12, n):
        slopes_window = slopes_all[t-12:t]
        for _ in range(EPOCHS_PER_STEP):
            y_pred = predict_gk(preds[-12:], slopes_window, w)
            grad = y_pred - y_all[t]
            w -= LEARNING_RATE * grad * np.sign(np.random.randn(14)) * 0.01
        final_pred = predict_gk(preds[-12:], slopes_window, w)
        preds.append(final_pred)
        weight_sets.append(deepcopy(w))

    preds = np.array(preds)


    pulled, l1_line = l2_l1_pull(preds, y_all[0], y_all[-1])
    center = np.polyval(np.polyfit(x_all, y_all, 2), x_all[:len(pulled)])
    tubed, upper, lower = tube_projection(pulled, center)
    final_adj = angle_strain(tubed)


    for idx in range(12, len(final_adj)):
        slopes_window = slopes_all[idx-12:idx]
        for _ in range(AFTER_TRAIN_EPOCHS):
            pred = predict_gk(final_adj[idx-12:idx], slopes_window, weight_sets[idx-12])
            grad = (pred - final_adj[idx])
            weight_sets[idx-12] -= LEARNING_RATE * grad * np.sign(np.random.randn(14)) * 0.01

    return {
        "raw_preds": preds.tolist(),
        "corrected_preds": final_adj.tolist(),
        "weight_sets": [w.tolist() for w in weight_sets],
        "x_all": x_all.tolist(),
        "l1_line": l1_line.tolist(),
        "tube_upper": upper.tolist(),
        "tube_lower": lower.tolist()
    }

# added different evaluation graph estimation
def evaluate(graphs):
    total_loss = 0
    count = 0
    for g in graphs:
        y_all = np.array(g["y"], dtype=float)
        slopes_all = np.array(g["slopes"], dtype=float)
        n = len(y_all)
        #assert n >= 13, "Each graph must have at least 13 points"
        preds = list(y_all[:12])
        w = np.zeros(14)
        for t in range(12, n):
            slopes_window = slopes_all[t-12:t]
            p = predict_gk(preds[-12:], slopes_window, w)
            preds.append(p)
        total_loss += np.mean((np.array(preds) - y_all)**2)
        count += 1
    return total_loss/count if count>0 else 0


def main():
    train_graphs = load_graphs("data/dynamic/train.json")
    print(train_graphs[0])
    val_graphs   = load_graphs("data/dynamic/val.json")
    test_graphs  = load_graphs("data/dynamic/test.json")

    out_dir = os.path.join(RESULTS_ROOT, f"spoke_tube_{int(time.time())}")
    safe_mkdir(out_dir)

    metrics = {"train_loss": [], "val_loss": 0, "test_loss":0}

    for idx, g in enumerate(train_graphs):
        print(f"Training graph {idx+1}/{len(train_graphs)}")
        result = train_single_graph(g)

        with open(os.path.join(out_dir,f"graph_{idx}_weights.json"), "w") as f:
            json.dump(result["weight_sets"], f)
        with open(os.path.join(out_dir,f"graph_{idx}_preds.json"), "w") as f:
            json.dump({
                "raw": result["raw_preds"],
                "corrected": result["corrected_preds"],
                "l1_line": result["l1_line"],
                "tube_upper": result["tube_upper"],
                "tube_lower": result["tube_lower"]
            }, f)

        plt.figure()
        x_all = result["x_all"]
        plt.plot(x_all, g["y"], 'k-', label="True")
        plt.plot(x_all[:len(result["raw_preds"])], result["raw_preds"], 'b--', label="Raw Pred")
        plt.plot(x_all[:len(result["corrected_preds"])], result["corrected_preds"], 'r-', label="Corrected")
        plt.plot(x_all[:len(result["l1_line"])], result["l1_line"], 'y--', label="L1 Line")
        plt.fill_between(x_all[:len(result["tube_upper"])], result["tube_lower"], result["tube_upper"], color='orange', alpha=0.3, label="Tube")
        plt.legend(); plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f"graph_{idx}_plot.png"))
        plt.close()

        train_loss = np.mean((np.array(result["corrected_preds"]) - np.array(g["y"][:len(result["corrected_preds"])]))**2)
        metrics["train_loss"].append(train_loss)

    metrics["val_loss"] = evaluate(val_graphs)
    metrics["test_loss"] = evaluate(test_graphs)

    with open(os.path.join(out_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Training complete. Results saved in: {out_dir}")
    print(f"Train loss: {np.mean(metrics['train_loss']):.6f}, Val loss: {metrics['val_loss']:.6f}, Test loss: {metrics['test_loss']:.6f}")

if __name__=="__main__":
    main()
