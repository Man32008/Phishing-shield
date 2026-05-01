"""
phishing_detector.py
====================
A beginner-friendly phishing URL detection system using Logistic Regression.
Built with scikit-learn for a university ML project demonstration.

Author : Student Project
Dataset: Hardcoded sample URLs for demonstration purposes
"""

# ── Imports ────────────────────────────────────────────────────────────────────
import re
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, classification_report
from sklearn.preprocessing import StandardScaler


# ── 1. Sample Dataset ──────────────────────────────────────────────────────────
# Each entry is (url, label) where label 1 = Phishing, 0 = Legitimate
SAMPLE_URLS = [
    # ── Phishing URLs ──────────────────────────────────────────────────────────
    ("http://paypal-login-verify.com/account/confirm", 1),
    ("http://secure-bankofamerica.login-now.com", 1),
    ("http://192.168.1.1/verify-account?user=admin@gmail.com", 1),
    ("http://amazon-security-alert.com/login/verify", 1),
    ("http://apple-id.verify-account-now.xyz/signin", 1),
    ("http://www.paypa1.com/login?redirect=bank", 1),
    ("http://ebay-account.login-secure.net/verify-identity", 1),
    ("http://update-your-banking-details.ru/login", 1),
    ("http://google-account-verify.tk/signin?email=user@gmail.com", 1),
    ("http://halifax-online.secure-login-bank.com/verify", 1),
    ("http://facebook-password-reset.suspicious-link.com/login", 1),
    ("http://netflixx.com/account/login@verify", 1),
    ("http://microsoft-verify.security-check.ml/login", 1),
    ("http://dropbox-login.phishing-example.org/signin", 1),
    ("http://wells.fargo.account.verify-bank-details.com/login", 1),

    # ── Legitimate URLs ────────────────────────────────────────────────────────
    ("https://www.google.com", 0),
    ("https://www.amazon.com/products", 0),
    ("https://github.com/user/repository", 0),
    ("https://www.wikipedia.org/wiki/Python", 0),
    ("https://stackoverflow.com/questions/tagged/python", 0),
    ("https://www.apple.com/iphone", 0),
    ("https://www.microsoft.com/en-us/windows", 0),
    ("https://www.linkedin.com/in/username", 0),
    ("https://www.youtube.com/watch?v=dQw4w9WgXcQ", 0),
    ("https://www.reddit.com/r/learnpython", 0),
    ("https://docs.python.org/3/library/re.html", 0),
    ("https://www.nytimes.com/section/technology", 0),
    ("https://www.bbc.com/news/technology", 0),
    ("https://www.paypal.com/myaccount/summary", 0),
    ("https://www.facebook.com/marketplace", 0),
]


# ── 2. Feature Extraction ───────────────────────────────────────────────────────
# Suspicious keywords commonly found in phishing URLs
SUSPICIOUS_KEYWORDS = ["login", "verify", "bank", "secure", "account",
                        "update", "confirm", "password", "signin", "alert",
                        "billing", "identity", "suspend"]


def extract_features(url: str) -> list:
    """
    Extract numerical features from a URL string.

    Features:
        1. url_length          – Total character length of the URL
        2. has_at_symbol       – 1 if '@' is present (common in phishing)
        3. dot_count           – Number of dots (subdomains inflate this)
        4. uses_https          – 1 if scheme is HTTPS, 0 otherwise
        5. suspicious_keywords – Count of suspicious words in the URL
        6. has_ip_address      – 1 if URL contains a raw IPv4 address
        7. subdomain_depth     – Number of subdomains (dots in hostname - 1)
        8. url_entropy         – Rough measure of character randomness

    Returns:
        list: A list of 8 numerical feature values.
    """
    url_lower = url.lower()

    # 1. Total URL length
    url_length = len(url)

    # 2. Presence of '@' symbol (attacker tricks browser into using left side)
    has_at_symbol = 1 if "@" in url else 0

    # 3. Total number of dots in the URL
    dot_count = url.count(".")

    # 4. Uses HTTPS (legitimate sites almost always do)
    uses_https = 1 if url_lower.startswith("https://") else 0

    # 5. Count how many suspicious keywords appear in the URL
    suspicious_keywords = sum(1 for kw in SUSPICIOUS_KEYWORDS if kw in url_lower)

    # 6. Detect raw IP address in URL (e.g., http://192.168.1.1/...)
    ip_pattern = re.compile(r"(\d{1,3}\.){3}\d{1,3}")
    has_ip_address = 1 if ip_pattern.search(url) else 0

    # 7. Subdomain depth – extract hostname and count dots
    try:
        hostname = re.findall(r"://([^/]+)", url)[0]
        subdomain_depth = hostname.count(".")
    except IndexError:
        subdomain_depth = 0

    # 8. URL character entropy (higher = more random / obfuscated)
    from collections import Counter
    counts = Counter(url)
    total = len(url)
    entropy = -sum((c / total) * np.log2(c / total) for c in counts.values()) if total > 0 else 0

    return [
        url_length,
        has_at_symbol,
        dot_count,
        uses_https,
        suspicious_keywords,
        has_ip_address,
        subdomain_depth,
        round(entropy, 4),
    ]


# ── 3. Build Dataset ───────────────────────────────────────────────────────────
def build_dataset(url_list: list):
    """Convert the list of (url, label) pairs into feature matrix X and labels y."""
    X = [extract_features(url) for url, _ in url_list]
    y = [label for _, label in url_list]
    return np.array(X), np.array(y)


# ── 4. Train Model ─────────────────────────────────────────────────────────────
def train_model(X, y):
    """
    Split data, scale features, train a Logistic Regression classifier,
    and print evaluation metrics.

    Returns:
        model   : Trained LogisticRegression object
        scaler  : Fitted StandardScaler (needed to transform new inputs)
    """
    # Split into 80 % training / 20 % testing
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Scale features so they are on a comparable range
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled  = scaler.transform(X_test)

    # Train Logistic Regression
    model = LogisticRegression(random_state=42, max_iter=1000)
    model.fit(X_train_scaled, y_train)

    # ── Evaluation ─────────────────────────────────────────────────────────────
    y_pred = model.predict(X_test_scaled)

    print("=" * 55)
    print("         MODEL EVALUATION RESULTS")
    print("=" * 55)
    print(f"  Accuracy  : {accuracy_score(y_test, y_pred):.2%}")
    print(f"  Precision : {precision_score(y_test, y_pred, zero_division=0):.2%}")
    print(f"  Recall    : {recall_score(y_test, y_pred, zero_division=0):.2%}")
    print()
    print("  Detailed Classification Report:")
    print(classification_report(y_test, y_pred,
                                 target_names=["Legitimate", "Phishing"],
                                 zero_division=0))
    print("=" * 55)

    return model, scaler


# ── 5. Predict Function ────────────────────────────────────────────────────────
def predict(url: str, model: LogisticRegression, scaler: StandardScaler) -> str:
    """
    Predict whether a given URL is phishing or legitimate.

    Args:
        url    : The URL string to classify.
        model  : Trained LogisticRegression model.
        scaler : Fitted StandardScaler from training.

    Returns:
        str: 'Phishing' or 'Legitimate'
    """
    features = np.array(extract_features(url)).reshape(1, -1)
    features_scaled = scaler.transform(features)
    prediction = model.predict(features_scaled)[0]
    confidence = model.predict_proba(features_scaled)[0]

    label = "Phishing" if prediction == 1 else "Legitimate"
    confidence_pct = confidence[prediction] * 100

    print(f"  URL        : {url}")
    print(f"  Prediction : {label}  (confidence: {confidence_pct:.1f}%)")
    print(f"  Features   : length={len(url)}, https={'yes' if url.startswith('https') else 'no'}, "
          f"dots={url.count('.')}, @={'yes' if '@' in url else 'no'}")
    print()

    return label


# ── 6. Main ────────────────────────────────────────────────────────────────────
def main():
    print("\n" + "=" * 55)
    print("   PHISHING URL DETECTION SYSTEM")
    print("   Using Logistic Regression (scikit-learn)")
    print("=" * 55 + "\n")

    # Build dataset and train model
    X, y = build_dataset(SAMPLE_URLS)
    model, scaler = train_model(X, y)

    # ── Demo predictions ───────────────────────────────────────────────────────
    test_urls = [
        "http://paypal-verify-account.login-secure.com/signin",   # likely phishing
        "https://www.github.com/explore",                          # likely legitimate
        "http://192.168.0.1/bank-login-verify?user=admin@test.com", # likely phishing
        "https://www.amazon.com/s?k=laptop",                       # likely legitimate
        "http://amazon-account-suspend-verify.ml/login",           # likely phishing
    ]

    print("\n" + "=" * 55)
    print("         SAMPLE PREDICTIONS")
    print("=" * 55 + "\n")

    for url in test_urls:
        predict(url, model, scaler)


if __name__ == "__main__":
    main()
