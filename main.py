import os
from conllu import parse_incr
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from collections import Counter, defaultdict

# -----------------------------
# CONFIGURATION
# -----------------------------

DATA_FOLDER = "ud_data"

FILES = {
    "English": "en_ewt-ud-train.conllu",
    "Hindi": "hi_hdtb-ud-train.conllu",
    "Spanish": "es_ancora-ud-train.conllu",
    "French": "fr_gsd-ud-train.conllu",
    "German": "de_gsd-ud-train.conllu",
    "Chinese": "zh_gsd-ud-train.conllu",
    "Russian": "ru_syntagrus-ud-train-a.conllu",
    "Arabic": "ar_padt-ud-train.conllu",
    "Turkish": "tr_imst-ud-train.conllu",
    "Japanese": "ja_gsd-ud-train.conllu"
}

# -----------------------------
# HELPER FUNCTIONS
# -----------------------------

def get_subtree_size(node_id, children_map):
    size = 1
    for child_id in children_map.get(node_id, []):
        size += get_subtree_size(child_id, children_map)
    return size

# -----------------------------
# CORE PROCESSING FUNCTION
# -----------------------------

def process_file(file_path):
    intervener_counts = []
    pos_counts = Counter()
    arity_counts = []
    phrase_lengths = []
    distance_list = []

    left = 0
    right = 0

    modifies_head = 0
    modifies_dep = 0
    modifies_other = 0

    total_dependencies = 0

    if not os.path.exists(file_path):
        print(f"Missing file: {file_path}")
        return None

    with open(file_path, "r", encoding="utf-8") as f:
        for sentence in parse_incr(f):
            tokens = [tok for tok in sentence if isinstance(tok["id"], int)]

            children = defaultdict(list)
            for tok in tokens:
                if tok["head"] is not None:
                    children[tok["head"]].append(tok["id"])

            for tok in tokens:
                head = tok["head"]
                if head is None or head == 0:
                    continue

                dep_id = tok["id"]
                head_id = head
                total_dependencies += 1

                # distance
                distance = abs(head_id - dep_id)
                distance_list.append(distance)

                # direction
                if dep_id < head_id:
                    left += 1
                else:
                    right += 1

                start = min(dep_id, head_id)
                end = max(dep_id, head_id)

                interveners = [
                    t for t in tokens
                    if start < t["id"] < end
                ]

                valid_interveners = [
                    iv for iv in interveners
                    if iv["upostag"] not in ["PUNCT", "SYM"]
                ]

                intervener_counts.append(len(valid_interveners))

                for iv in valid_interveners:
                    pos_counts[iv["upostag"]] += 1
                    arity_counts.append(len(children[iv["id"]]))
                    phrase_lengths.append(get_subtree_size(iv["id"], children))

                    if iv["head"] == head_id:
                        modifies_head += 1
                    elif iv["head"] == dep_id:
                        modifies_dep += 1
                    else:
                        modifies_other += 1

    return {
        "intervener_counts": intervener_counts,
        "pos_counts": pos_counts,
        "arity_counts": arity_counts,
        "phrase_lengths": phrase_lengths,
        "distances": distance_list,
        "total_dependencies": total_dependencies,
        "left": left,
        "right": right,
        "mod_head": modifies_head,
        "mod_dep": modifies_dep,
        "mod_other": modifies_other
    }

# -----------------------------
# PROCESS ALL LANGUAGES
# -----------------------------

all_intervener_counts = []
all_arity_counts = []
all_phrase_lengths = []
all_distances = []
all_pos_counts = Counter()

language_stats = []

for lang, filename in FILES.items():
    print(f"Processing {lang}...")
    path = os.path.join(DATA_FOLDER, filename)

    result = process_file(path)
    if result is None:
        continue

    all_intervener_counts.extend(result["intervener_counts"])
    all_arity_counts.extend(result["arity_counts"])
    all_phrase_lengths.extend(result["phrase_lengths"])
    all_distances.extend(result["distances"])
    all_pos_counts.update(result["pos_counts"])

    language_stats.append({
        "Language": lang,
        "Dependencies": result["total_dependencies"],
        "Avg Interveners": sum(result["intervener_counts"]) / max(len(result["intervener_counts"]),1),
        "Avg Arity": sum(result["arity_counts"]) / max(len(result["arity_counts"]),1),
        "Avg Phrase Length": sum(result["phrase_lengths"]) / max(len(result["phrase_lengths"]),1),
        "Avg Distance": sum(result["distances"]) / max(len(result["distances"]),1),
        "Left": result["left"],
        "Right": result["right"],
        "Modifies Head": result["mod_head"],
        "Modifies Dep": result["mod_dep"],
        "Modifies Other": result["mod_other"]
    })

# -----------------------------
# DATAFRAMES
# -----------------------------

df_lang = pd.DataFrame(language_stats)

df_pos = pd.DataFrame(all_pos_counts.items(), columns=["POS", "Count"])
df_pos["Normalized"] = df_pos["Count"] / df_pos["Count"].sum()

# Re-added detailed DataFrames
df_intervener = pd.DataFrame({"Interveners": all_intervener_counts})
df_arity = pd.DataFrame({"Arity": all_arity_counts})
df_length = pd.DataFrame({"Phrase Length": all_phrase_lengths})

# -----------------------------
# SAVE CSV
# -----------------------------

df_lang.to_csv("language_summary.csv", index=False)
df_pos.to_csv("pos_distribution.csv", index=False)

# Re-added CSV saves
df_intervener.to_csv("intervener_counts.csv", index=False)
df_arity.to_csv("arity_counts.csv", index=False)
df_length.to_csv("phrase_length_counts.csv", index=False)

# -----------------------------
# PLOTTING
# -----------------------------

sns.set_theme(style="whitegrid")

# 1. Distance distribution
plt.figure()
sns.histplot(all_distances, bins=20)
plt.title("Dependency Distance Distribution")
plt.xlabel("Distance")
plt.ylabel("Frequency")
plt.savefig("distance_distribution.png")
plt.close()

# 2. POS normalized
plt.figure(figsize=(10,6))
df_pos_sorted = df_pos.sort_values(by="Normalized", ascending=False)
sns.barplot(data=df_pos_sorted, x="POS", y="Normalized")
plt.xticks(rotation=45)
plt.title("Normalized POS Distribution")
plt.savefig("pos_normalized.png")
plt.close()

# 3. Avg interveners per language
plt.figure()
sns.barplot(data=df_lang, x="Language", y="Avg Interveners")
plt.xticks(rotation=45)
plt.title("Avg Interveners per Language")
plt.savefig("lang_interveners.png")
plt.close()

# 4. Avg distance per language
plt.figure()
sns.barplot(data=df_lang, x="Language", y="Avg Distance")
plt.xticks(rotation=45)
plt.title("Avg Dependency Distance per Language")
plt.savefig("lang_distance.png")
plt.close()

# 5. Direction plot
plt.figure()
direction_totals = pd.DataFrame({
    "Direction": ["Left", "Right"],
    "Count": [df_lang["Left"].sum(), df_lang["Right"].sum()]
})
sns.barplot(data=direction_totals, x="Direction", y="Count")
plt.title("Dependency Direction Distribution")
plt.savefig("direction.png")
plt.close()

# 6. Re-added: Intervener count distribution (excluding zeros for better visibility of actual interveners)
plt.figure()
sns.histplot(df_intervener[df_intervener["Interveners"] > 0]["Interveners"], bins=15, discrete=True)
plt.title("Distribution of Interveners per Dependency (Excluding 0)")
plt.xlabel("Number of Interveners")
plt.ylabel("Frequency")
plt.savefig("intervener_distribution.png")
plt.close()

# 7. Re-added: Arity distribution
plt.figure()
sns.histplot(df_arity["Arity"], bins=10, discrete=True)
plt.title("Arity Distribution of Interveners")
plt.xlabel("Arity")
plt.ylabel("Frequency")
plt.savefig("arity_distribution.png")
plt.close()

# 8. Re-added: Phrase Length Distribution
plt.figure()
sns.histplot(df_length["Phrase Length"], bins=15, discrete=True)
plt.title("Phrase Length (Subtree Size) of Interveners")
plt.xlabel("Phrase Length in Tokens")
plt.ylabel("Frequency")
plt.savefig("phrase_length_distribution.png")
plt.close()

print("\nDone. All outputs saved.")