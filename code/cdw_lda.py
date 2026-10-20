"""LDA topic modeling of Clinical Data Warehouse literature (136 Scopus abstracts, 2021-2026).

Reproduces the reported four-topic model exactly and adds the robustness analyses
(multi-seed perplexity and C_V coherence, Friedman/Wilcoxon tests, convergence and
topic stability).

Usage: python cdw_lda.py <path/to/136_Full.csv> <output_dir>
"""
import json
import os
import re
import sys
from itertools import combinations

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import nltk
import numpy as np
import pandas as pd
from gensim.corpora import Dictionary
from gensim.models.coherencemodel import CoherenceModel
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from scipy.optimize import linear_sum_assignment
from scipy.stats import friedmanchisquare, wilcoxon
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import adjusted_rand_score
from wordcloud import WordCloud

for pkg in ['punkt', 'punkt_tab', 'stopwords', 'wordnet', 'omw-1.4']:
    nltk.download(pkg, quiet=True)

CSV_PATH, OUT_DIR = sys.argv[1], sys.argv[2]
os.makedirs(OUT_DIR, exist_ok=True)

K = 4                       # selected number of topics
K_RANGE = range(2, 8)
SEED = 42                   # seed of the reported model
SEEDS = [SEED] + list(range(1, 10))
MAX_ITER = 50
TOP_N = 10
# scikit-learn's batch E-step splits documents into one chunk per job and seeds each
# chunk identically, so results depend on n_jobs. The reported model was fitted with
# n_jobs=-1 on a 12-thread machine; fixing n_jobs=12 reproduces it on any hardware.
N_JOBS = 12

# ---------------------------------------------------------------------------
# 1. Data
# ---------------------------------------------------------------------------
df = pd.read_csv(CSV_PATH)
df['Abstract'] = df['Abstract'].astype(str).str.strip()
df = df[~df['Abstract'].isin(['', 'nan', '[No abstract available]'])].reset_index(drop=True)
print(f'Documents: {len(df)}')

# ---------------------------------------------------------------------------
# 2. Preprocessing
# ---------------------------------------------------------------------------
# Matched on the raw token and on its lemma, so only base forms are listed.
# Clinical terms (patient, warehouse, cdw, hospital, cohort, opioid, ...) are kept.
DOMAIN_STOPWORDS = {
    # generic research terms
    'study', 'research', 'paper', 'result', 'data', 'analysis',
    'method', 'conclusion', 'abstract', 'introduction', 'literature',
    'discuss', 'finding', 'propose', 'suggest',
    # structured-abstract headers and generic study descriptors
    'background', 'objective', 'aim', 'purpose', 'design', 'setting',
    'methodology', 'overview', 'summary', 'scope', 'rationale',
    'context', 'review', 'article',
    # research verbs that do not discriminate topics
    'use', 'used', 'using', 'based', 'approach', 'apply', 'applied',
    'perform', 'performed', 'conduct', 'conducted', 'obtain', 'obtained',
    'include', 'included', 'exclude', 'excluded', 'inclusion', 'exclusion',
    'identify', 'identified', 'determine', 'determined',
    'evaluate', 'evaluated', 'assess', 'assessed', 'examine', 'examined',
    'investigate', 'investigated', 'analyze', 'analyse', 'analyzed',
    'calculate', 'calculated', 'collect', 'collected',
    'develop', 'developed', 'implement', 'implemented',
    'provide', 'provided', 'consider', 'considered', 'require', 'required',
    'utilize', 'utilized', 'generate', 'generated', 'create', 'created',
    'define', 'defined', 'describe', 'described', 'present', 'presented',
    'report', 'reported', 'observe', 'observed', 'show', 'showed', 'shown',
    'demonstrate', 'demonstrated', 'indicate', 'indicated',
    'reveal', 'revealed', 'estimate', 'estimated', 'measure', 'measured',
    'compare', 'compared', 'comparison',
    # statistical / quantitative fillers
    'significant', 'significantly', 'higher', 'lower', 'high', 'low',
    'increase', 'increased', 'decrease', 'decreased', 'number', 'total',
    'rate', 'ratio', 'proportion', 'percentage', 'percent', 'mean',
    'median', 'average', 'range', 'value', 'level', 'score', 'count',
    'frequency', 'associated', 'association', 'correlation', 'group',
    'sample', 'size', 'overall', 'respectively', 'versus', 'characteristic',
    # vague qualifiers and connectives
    'within', 'across', 'among', 'toward', 'however', 'therefore', 'thus',
    'whether', 'furthermore', 'moreover', 'additionally', 'various',
    'different', 'several', 'multiple', 'single', 'common', 'general',
    'specific', 'particular', 'particularly', 'specifically', 'potential',
    'important', 'relevant', 'available', 'current', 'recent', 'novel',
    'new', 'existing', 'similar', 'related', 'individual', 'status',
    'process', 'format', 'standard', 'field',
    # time markers
    'year', 'month', 'week', 'day', 'time', 'period', 'first', 'second',
    'third', 'prior', 'previous', 'subsequent',
    # Scopus/publisher boilerplate at the end of abstracts
    'elsevier', 'springer', 'wiley', 'taylor', 'francis', 'sage',
    'copyright', 'right', 'reserved', 'author', 'ltd', 'inc',
    'publisher', 'published', 'license', 'licensee', 'mdpi', 'bv',
}

lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english')) | DOMAIN_STOPWORDS


def preprocess(text):
    text = text.lower()
    text = re.sub(r'http\S+|www\S+', '', text)
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    tokens = []
    for token in word_tokenize(text):
        if token in stop_words or len(token) <= 2:
            continue
        lemma = lemmatizer.lemmatize(token)
        if lemma in stop_words or len(lemma) <= 2:
            continue
        tokens.append(lemma)
    return tokens


df['tokens'] = df['Abstract'].apply(preprocess)
vectorizer = CountVectorizer(max_features=5000, min_df=2, max_df=0.8, stop_words='english')
dtm = vectorizer.fit_transform(df['tokens'].str.join(' '))
vocab = vectorizer.get_feature_names_out()
print(f'Document-term matrix: {dtm.shape[0]} x {dtm.shape[1]}')

# Coherence reference corpus restricted to the modeled vocabulary
vocab_set = set(vocab)
texts = [[t for t in toks if t in vocab_set] for toks in df['tokens']]
dictionary = Dictionary(texts)


def fit_lda(k, seed, max_iter=MAX_ITER):
    return LatentDirichletAllocation(n_components=k, learning_method='batch', max_iter=max_iter,
                                     random_state=seed, n_jobs=N_JOBS).fit(dtm)


def top_words(model, n=TOP_N):
    return [[vocab[i] for i in comp.argsort()[::-1][:n]] for comp in model.components_]


def coherence_cv(model):
    return CoherenceModel(topics=top_words(model), texts=texts, dictionary=dictionary,
                          coherence='c_v', topn=TOP_N, processes=1).get_coherence()


def matched_similarity(a, b):
    """Mean cosine similarity of topic-word distributions after optimal topic matching."""
    a = a / np.linalg.norm(a, axis=1, keepdims=True)
    b = b / np.linalg.norm(b, axis=1, keepdims=True)
    sim = a @ b.T
    rows, cols = linear_sum_assignment(-sim)
    return sim[rows, cols].mean()


# ---------------------------------------------------------------------------
# 3. Number of topics over multiple seeds
# ---------------------------------------------------------------------------
records, components, labels = [], {}, {}
for k in K_RANGE:
    for seed in SEEDS:
        m = fit_lda(k, seed)
        components[(k, seed)] = m.components_
        labels[(k, seed)] = m.transform(dtm).argmax(1)
        records.append({'k': k, 'seed': seed, 'perplexity': m.perplexity(dtm),
                        'c_v': coherence_cv(m), 'n_iter': m.n_iter_})
    print(f'k={k} done', flush=True)
scores = pd.DataFrame(records)
scores.to_csv(os.path.join(OUT_DIR, 'k_selection_scores.csv'), index=False)
summary = scores.groupby('k')[['perplexity', 'c_v']].agg(['mean', 'std'])
print(summary.round(4).to_string())

# Stability: agreement between every pair of seeds
stability = []
for k in K_RANGE:
    pairs = list(combinations(SEEDS, 2))
    sims = [matched_similarity(components[(k, a)], components[(k, b)]) for a, b in pairs]
    aris = [adjusted_rand_score(labels[(k, a)], labels[(k, b)]) for a, b in pairs]
    stability.append({'k': k, 'topic_sim_mean': np.mean(sims), 'topic_sim_std': np.std(sims),
                      'ari_mean': np.mean(aris), 'ari_std': np.std(aris)})
stability = pd.DataFrame(stability)
stability.to_csv(os.path.join(OUT_DIR, 'stability.csv'), index=False)
print(stability.round(3).to_string())

# Friedman test across k (seeds as blocks), then Holm-corrected Wilcoxon
# signed-rank tests of K against every alternative
friedman, wilcox = {}, []
for metric in ['perplexity', 'c_v']:
    wide = scores.pivot(index='seed', columns='k', values=metric)
    stat, p = friedmanchisquare(*[wide[k] for k in K_RANGE])
    friedman[metric] = {'chi2': stat, 'p': p}
    rows = []
    for k in K_RANGE:
        if k != K:
            res = wilcoxon(wide[K], wide[k])
            rows.append({'metric': metric, 'comparison': f'k={K} vs k={k}',
                         'median_diff': float(np.median(wide[K] - wide[k])),
                         'W': res.statistic, 'p': res.pvalue})
    rows = pd.DataFrame(rows).sort_values('p').reset_index(drop=True)
    rows['p_holm'] = np.minimum(1, np.maximum.accumulate(rows['p'] * (len(rows) - rows.index)))
    wilcox.append(rows)
wilcox = pd.concat(wilcox, ignore_index=True)
wilcox.to_csv(os.path.join(OUT_DIR, 'wilcoxon.csv'), index=False)
print('Friedman:', friedman)
print(wilcox.round(4).to_string())

# ---------------------------------------------------------------------------
# 4. Reported model
# ---------------------------------------------------------------------------
lda = fit_lda(K, SEED)

# Convergence: perplexity after each EM iteration (batch EM is deterministic)
conv = [fit_lda(K, SEED, max_iter=i).perplexity(dtm) for i in range(1, MAX_ITER + 1)]
pd.DataFrame({'iteration': range(1, MAX_ITER + 1), 'perplexity': conv}).to_csv(
    os.path.join(OUT_DIR, 'convergence.csv'), index=False)

topics = top_words(lda)
theta = lda.transform(dtm)
topic_cols = [f'Topic_{i + 1}' for i in range(K)]
theta_df = pd.DataFrame(theta, columns=topic_cols)
df['dominant_topic'] = theta.argmax(1) + 1
dominant_counts = df['dominant_topic'].value_counts().sort_index()

rep_docs = []
for t in range(K):
    i = int(theta[:, t].argmax())
    rep_docs.append({'topic': t + 1, 'title': df.loc[i, 'Title'], 'source': df.loc[i, 'Source title'],
                     'year': int(df.loc[i, 'Year']), 'p': round(float(theta[i, t]), 3)})

yearly = pd.concat([df[['Year']], theta_df], axis=1).groupby('Year').mean()
corr = theta_df.corr()

pd.concat([df[['Title', 'Year', 'Source title', 'dominant_topic']], theta_df], axis=1).to_csv(
    os.path.join(OUT_DIR, 'document_topic_results.csv'), index=False)
yearly.to_csv(os.path.join(OUT_DIR, 'yearly_prevalence.csv'))
corr.to_csv(os.path.join(OUT_DIR, 'topic_correlation.csv'))

report = {
    'n_docs': int(dtm.shape[0]), 'vocab_size': int(dtm.shape[1]), 'k': K, 'seed': SEED,
    'n_jobs': N_JOBS, 'perplexity': lda.perplexity(dtm), 'c_v': coherence_cv(lda),
    'n_iter': int(lda.n_iter_),
    'convergence_last5_rel_change': float(abs(conv[-1] - conv[-6]) / conv[-6]),
    'friedman': friedman,
    'topics': {f'Topic {i + 1}': w for i, w in enumerate(topics)},
    'dominant_counts': {f'Topic {t}': int(c) for t, c in dominant_counts.items()},
    'representative_docs': rep_docs,
}
with open(os.path.join(OUT_DIR, 'report.json'), 'w', encoding='utf-8') as f:
    json.dump(report, f, indent=2, default=float)
print(json.dumps(report, indent=2, default=float))
print('Yearly prevalence:\n', yearly.round(3).to_string())
print('Topic correlation:\n', corr.round(3).to_string())

# ---------------------------------------------------------------------------
# 5. Figures
# ---------------------------------------------------------------------------
plt.rcParams.update({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False})
COLORS = ['#2a6fdb', '#e07a1f', '#2a9d5c', '#c0392b']
ks = list(K_RANGE)
seed_scores = scores[scores['seed'] == SEED].set_index('k')

fig, axes = plt.subplots(1, 2, figsize=(7, 2.7))
for ax, col, label in [(axes[0], 'perplexity', 'Perplexity'),
                       (axes[1], 'c_v', 'Coherence $C_V$')]:
    ax.errorbar(ks, summary[(col, 'mean')], yerr=summary[(col, 'std')], fmt='o-', capsize=4,
                color='#2a6fdb', linewidth=1.8, label=f'Mean $\\pm$ SD ({len(SEEDS)} seeds)')
    ax.plot(ks, seed_scores.loc[ks, col], 's--', color='#7f8c8d', linewidth=1, markersize=4,
            label=f'Reported model (seed {SEED})')
    ax.axvline(K, color='#bbbbbb', linestyle=':', linewidth=1)
    ax.set_xlabel('Number of topics ($k$)')
    ax.set_ylabel(label)
    ax.set_xticks(ks)
axes[0].legend(frameon=False, fontsize=7.5, loc='upper left')
fig.tight_layout()
fig.savefig(os.path.join(OUT_DIR, 'optimal_topics.png'), dpi=300)
plt.close(fig)

fig, ax = plt.subplots(figsize=(5, 3.2))
ax.plot(range(1, MAX_ITER + 1), conv, color='#2a6fdb', linewidth=1.8)
ax.set_xlabel('EM iteration')
ax.set_ylabel('Perplexity')
fig.tight_layout()
fig.savefig(os.path.join(OUT_DIR, 'convergence.png'), dpi=300)
plt.close(fig)

fig, axes = plt.subplots(2, 2, figsize=(8, 6.4))
for t, ax in enumerate(axes.flatten()):
    comp = lda.components_[t]
    idx = comp.argsort()[::-1][:TOP_N]
    wc = WordCloud(width=500, height=300, background_color='white', colormap='viridis',
                   random_state=0).generate_from_frequencies({vocab[i]: float(comp[i]) for i in idx})
    ax.imshow(wc, interpolation='bilinear')
    ax.set_title(f'Topic {t + 1}', fontweight='bold')
    ax.axis('off')
fig.tight_layout()
fig.savefig(os.path.join(OUT_DIR, 'topic_wordclouds.png'), dpi=300)
plt.close(fig)

fig, ax = plt.subplots(figsize=(6, 2.6))
for t in range(K):
    ax.plot(yearly.index, yearly[topic_cols[t]], 'o-', color=COLORS[t], linewidth=1.8,
            label=f'Topic {t + 1}')
ax.set_xlabel('Publication year')
ax.set_ylabel('Mean topic proportion')
ax.set_xticks(yearly.index)
ax.legend(frameon=False, ncol=4, loc='upper center', bbox_to_anchor=(0.5, 1.2))
fig.tight_layout()
fig.savefig(os.path.join(OUT_DIR, 'topic_trends.png'), dpi=300)
plt.close(fig)

fig, ax = plt.subplots(figsize=(4.2, 3.4))
im = ax.imshow(corr.values, cmap='RdBu_r', vmin=-1, vmax=1)
names = [f'T{i + 1}' for i in range(K)]
ax.set_xticks(range(K), names)
ax.set_yticks(range(K), names)
for i in range(K):
    for j in range(K):
        ax.text(j, i, f'{corr.values[i, j]:+.2f}', ha='center', va='center', fontsize=9,
                color='white' if abs(corr.values[i, j]) > 0.6 else 'black')
ax.spines[:].set_visible(False)
fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label='Pearson $r$')
fig.tight_layout()
fig.savefig(os.path.join(OUT_DIR, 'topic_network.png'), dpi=300, bbox_inches='tight')
plt.close(fig)
print('Figures written to', OUT_DIR)
