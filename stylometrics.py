from __future__ import annotations

import re
from collections import Counter

import numpy as np
from scipy import sparse
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler


def preprocess_agresivo(text: str) -> str:
    """Limpieza SOTA para OCR severamente dañado: Ablación de ruido."""
    if not isinstance(text, str):
        return ''
    text = text.lower()
    # 1. Unir guiones a final de línea
    text = re.sub(r'-\s*\n\s*', '', text)
    # 2. Eliminar números
    text = re.sub(r'\d+', ' ', text)
    # 3. Solo letras del español + espacios
    text = re.sub(r'[^a-záéíóúñüç\s]', ' ', text)
    # 4. Eliminar consonantes flotantes
    text = re.sub(r'\b(?![aeiouyáéíóú])[a-záéíóúñüç]\b', ' ', text)
    # 5. Colapsar espacios
    text = re.sub(r'\s+', ' ', text).strip()
    return text


class StylometricFeatureExtractor(BaseEstimator, TransformerMixin):
    """Extract Yule's K and standardize it for downstream linear models."""

    def __init__(self):
        self.scaler = StandardScaler()

    def fit(self, X, y=None):
        values = self._compute_values(X)
        self.scaler.fit(values)
        return self

    @staticmethod
    def _yule_k(text: str) -> float:
        tokens = re.findall(r"[^\W\d_]+", str(text).lower())
        if not tokens:
            return 0.0

        freqs = Counter(tokens).values()
        n_tokens = len(tokens)
        sum_sq = sum(freq * freq for freq in freqs)
        return 10000.0 * max(sum_sq - n_tokens, 0.0) / (n_tokens ** 2)

    def _compute_values(self, X):
        return np.array([self._yule_k(text) for text in X], dtype=float).reshape(-1, 1)

    def transform(self, X):
        values = self._compute_values(X)
        scaled = self.scaler.transform(values)
        return sparse.csr_matrix(scaled)


# Backward-compatible alias for older notebook/script references.
StylometricTransformer = StylometricFeatureExtractor


class RichStylometricExtractor(BaseEstimator, TransformerMixin):
    """Stylometric features ricas, computadas sobre TEXTO CRUDO.

    A diferencia de StylometricFeatureExtractor (1 dim), entrega 14 dimensiones
    independientes del vocabulario que cambian sistemáticamente con el siglo:
    densidad de puntuación, mayúsculas, dígitos, vocales acentuadas, "s larga",
    longitud media de palabra y oración, type-token ratio, etc.

    Importante: el input debe ser texto CRUDO (sin preprocesar). La señal está
    en la puntuación, mayúsculas y caracteres OCR-específicos que el preprocess
    agresivo elimina.
    """

    N_FEATURES = 14
    PUNCT_CHARS = set('.,;:!?¡¿«»()[]{}—–-"\'\u2018\u2019\u201c\u201d')

    def __init__(self):
        self.scaler = StandardScaler()

    @staticmethod
    def _features(text):
        if not isinstance(text, str) or len(text) == 0:
            return [0.0] * RichStylometricExtractor.N_FEATURES

        n_chars = max(1, len(text))
        text_lower = text.lower()

        tokens = re.findall(r"[^\W\d_]+", text_lower)
        n_tokens = len(tokens)
        unique_tokens = len(set(tokens)) if tokens else 0
        n_tokens_safe = max(1, n_tokens)

        # Yule K (riqueza léxica)
        if tokens:
            freqs = list(Counter(tokens).values())
            sum_sq = sum(f * f for f in freqs)
            yule_k = 10000.0 * max(sum_sq - n_tokens, 0.0) / (n_tokens ** 2)
        else:
            yule_k = 0.0

        # Length / lexical
        avg_word_len = sum(len(t) for t in tokens) / n_tokens_safe if tokens else 0.0
        ttr = unique_tokens / n_tokens_safe

        # Char-level ratios sobre el RAW (preserva puntuación / case)
        n_upper = sum(1 for c in text if c.isupper())
        n_digit = sum(1 for c in text if c.isdigit())
        n_punct = sum(1 for c in text if c in RichStylometricExtractor.PUNCT_CHARS)
        n_vowel = sum(1 for c in text_lower if c in 'aeiouáéíóú')
        n_accented = sum(1 for c in text_lower if c in 'áéíóúñü')
        n_long_s = text.count('ſ')
        n_space = text.count(' ')

        # Sentence stats
        sentences = [s for s in re.split(r'[.!?]+', text) if s.strip()]
        n_sent = max(1, len(sentences))
        avg_sent_len_tokens = n_tokens / n_sent if n_tokens else 0.0

        return [
            yule_k,
            avg_word_len,
            ttr,
            n_upper / n_chars,
            n_digit / n_chars,
            n_punct / n_chars,
            n_vowel / n_chars,
            n_accented / n_chars,
            n_long_s / n_chars,
            n_space / n_chars,
            avg_sent_len_tokens,
            float(np.log1p(n_chars)),
            float(np.log1p(n_tokens_safe)),
            unique_tokens / max(1.0, n_chars / 100.0),
        ]

    def _compute(self, X):
        return np.array([self._features(t) for t in X], dtype=float)

    def fit(self, X, y=None):
        self.scaler.fit(self._compute(X))
        return self

    def transform(self, X):
        return sparse.csr_matrix(self.scaler.transform(self._compute(X)))


class RawCharCleanWordLogRegPipeline:
    """Pipeline final para accuracy con vistas mixtas sobre raw/clean text.

    Usa al menos:
    - char TF-IDF sobre raw
    - word TF-IDF sobre clean
    - stylometrics sobre raw

    Opcionalmente puede incluir un segundo canal `char_wb` sobre raw para
    capturar patrones internos de palabra sin perder la señal de espacios,
    puntuación y OCR del canal `char`.
    """

    def __init__(self, raw_char_vec, clean_word_vec, styl, clf, raw_char_wb_vec=None):
        self.raw_char_vec = raw_char_vec
        self.raw_char_wb_vec = raw_char_wb_vec
        self.clean_word_vec = clean_word_vec
        self.styl = styl
        self.clf = clf
        self.classes_ = clf.classes_

    def _featurize(self, raw_texts):
        raw_list = list(raw_texts)
        clean_list = [preprocess_agresivo(t) for t in raw_list]
        blocks = [self.raw_char_vec.transform(raw_list)]
        if self.raw_char_wb_vec is not None:
            blocks.append(self.raw_char_wb_vec.transform(raw_list))
        blocks.extend(
            [
                self.clean_word_vec.transform(clean_list),
                self.styl.transform(raw_list),
            ]
        )
        return sparse.hstack(blocks, format="csr")

    def predict(self, raw_texts):
        return self.clf.predict(self._featurize(raw_texts))

    def decision_function(self, raw_texts):
        return self.clf.decision_function(self._featurize(raw_texts))


class EnsemblePipeline:
    """Pipeline ensemble: LinearSVC + LogisticRegression + ComplementNB.

    Combina los tres modelos por z-scores promediados con pesos sintonizados
    contra el hold-out. Acepta TEXTO CRUDO como input — la limpieza ocurre
    internamente para que la API sea consistente y serializable.
    """

    def __init__(self, char_vec, word_vec, styl, nb_char_vec, nb_word_vec,
                 svc, lr, nb, weights):
        self.char_vec = char_vec
        self.word_vec = word_vec
        self.styl = styl
        self.nb_char_vec = nb_char_vec
        self.nb_word_vec = nb_word_vec
        self.svc = svc
        self.lr = lr
        self.nb = nb
        self.weights = tuple(weights)
        self.classes_ = svc.classes_

    @staticmethod
    def _zscore(scores):
        s = np.asarray(scores, dtype=float)
        mu = s.mean(axis=1, keepdims=True)
        sd = s.std(axis=1, keepdims=True) + 1e-9
        return (s - mu) / sd

    def _featurize_main(self, clean_texts, raw_texts):
        return sparse.hstack([
            self.char_vec.transform(clean_texts),
            self.word_vec.transform(clean_texts),
            self.styl.transform(raw_texts),
        ], format='csr')

    def _featurize_nb(self, clean_texts):
        return sparse.hstack([
            self.nb_char_vec.transform(clean_texts),
            self.nb_word_vec.transform(clean_texts),
        ], format='csr')

    def predict(self, raw_texts):
        raw_list = list(raw_texts)
        clean_list = [preprocess_agresivo(t) for t in raw_list]
        X_main = self._featurize_main(clean_list, raw_list)
        X_nb = self._featurize_nb(clean_list)
        z_svc = self._zscore(self.svc.decision_function(X_main))
        z_lr = self._zscore(self.lr.decision_function(X_main))
        z_nb = self._zscore(self.nb.predict_log_proba(X_nb))
        w_svc, w_lr, w_nb = self.weights
        ensemble = w_svc * z_svc + w_lr * z_lr + w_nb * z_nb
        return self.classes_[ensemble.argmax(axis=1)]
