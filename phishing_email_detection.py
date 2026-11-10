# ==========================================================
# PHISHING EMAIL DETECTION MODEL
# Machine Learning using Scikit-learn
# Algorithm: Multinomial Naive Bayes
#
# Install dependencies:
# pip install pandas scikit-learn matplotlib scipy
#
# Run:
# python phishing_email_detection.py
# ==========================================================

import re

import pandas as pd
import matplotlib.pyplot as plt

from scipy.sparse import csr_matrix, hstack
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB


# STEP 1: CREATE A DEMONSTRATION DATASET
# Label 1 = Phishing, Label 0 = Safe / legitimate.
# Replace or expand this small sample with a larger, labelled dataset
# before using the model for meaningful real-world evaluation.

data = {
    "email": [
        # Phishing emails
        "Congratulations! You won a free iPhone. Click here to claim your prize.",
        "Your bank account is suspended. Verify your password immediately.",
        "You have won a lottery. Send your bank details to receive money.",
        "Urgent! Your account will be blocked. Click this suspicious link now.",
        "Your ATM card is blocked. Verify your PIN at http://fakebank-login.com",
        "Claim your free gift now. Limited time offer.",
        "Your account has unusual activity. Login immediately to secure it.",
        "You won a cash prize. Provide your credit card details.",
        "Update your KYC immediately to avoid account suspension.",
        "Your package is waiting. Pay a small fee at http://delivery-check.xyz",
        "Click http://192.168.1.10/login to verify your account.",
        "Your password expires today. Confirm your credentials immediately.",

        # Safe / legitimate emails
        "Meeting scheduled for tomorrow at 10 AM.",
        "Please find the attached project report.",
        "Your payment has been received successfully.",
        "Thank you for attending today's college seminar.",
        "The assignment submission deadline is Friday.",
        "Your monthly electricity bill is available in your account.",
        "The team meeting has been moved to Monday.",
        "Please review the attached presentation.",
        "Your order has been shipped successfully.",
        "The examination timetable has been published.",
        "Thank you for contacting our customer support team.",
        "Your appointment is confirmed for tomorrow.",
    ],
    "label": [
        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    ],
}

df = pd.DataFrame(data)

print("\n========== EMAIL DATASET ==========")
print(df.to_string(index=False))


# STEP 2: DEFINE SUSPICIOUS KEYWORDS

SUSPICIOUS_KEYWORDS = [
    "urgent",
    "verify",
    "password",
    "account suspended",
    "blocked",
    "free",
    "winner",
    "lottery",
    "claim",
    "bank details",
    "credit card",
    "login",
    "immediately",
    "click here",
    "kyc",
    "pin",
]


# STEP 3: EXTRACT URL AND CONTENT FEATURES

def extract_features(email):
    """Return simple numerical features extracted from an email."""
    email = email.lower()

    urls = re.findall(
        r"https?://[^\s]+|www\.[^\s]+",
        email,
    )

    keyword_count = sum(
        email.count(keyword)
        for keyword in SUSPICIOUS_KEYWORDS
    )

    has_ip_address_url = int(
        bool(
            re.search(
                r"https?://(?:\d{1,3}\.){3}\d{1,3}",
                email,
            )
        )
    )

    has_url_shortener = int(
        any(
            domain in email
            for domain in ("bit.ly", "tinyurl.com", "t.co")
        )
    )

    https_count = len(re.findall(r"https://", email))

    return [
        len(urls),
        keyword_count,
        has_ip_address_url,
        has_url_shortener,
        https_count,
    ]


# STEP 4: SEPARATE TEXT, LABELS, AND EXTRACTED FEATURES

X_text = df["email"]
y = df["label"]

X_features = [extract_features(email) for email in X_text]


# STEP 5: SPLIT DATA INTO TRAINING AND TESTING SETS

(
    X_train,
    X_test,
    y_train,
    y_test,
    features_train,
    features_test,
) = train_test_split(
    X_text,
    y,
    X_features,
    test_size=0.25,
    random_state=42,
    stratify=y,
)


# STEP 6: CONVERT EMAIL TEXT INTO NUMERICAL FEATURES

vectorizer = CountVectorizer()

X_train_vectorized = vectorizer.fit_transform(X_train)
X_test_vectorized = vectorizer.transform(X_test)


# STEP 7: COMBINE TEXT FEATURES AND EMAIL FEATURES

X_train_final = hstack(
    [X_train_vectorized, csr_matrix(features_train)]
)

X_test_final = hstack(
    [X_test_vectorized, csr_matrix(features_test)]
)


# STEP 8: TRAIN THE MULTINOMIAL NAIVE BAYES MODEL

model = MultinomialNB()
model.fit(X_train_final, y_train)


# STEP 9: PREDICT THE TEST SET

y_pred = model.predict(X_test_final)


# STEP 10: EVALUATE THE MODEL

accuracy = accuracy_score(y_test, y_pred)

print("\n========== MODEL EVALUATION ==========")
print(f"Model Accuracy: {accuracy * 100:.2f}%")

print("\nClassification Report:")
print(
    classification_report(
        y_test,
        y_pred,
        labels=[0, 1],
        target_names=["Safe Email", "Phishing Email"],
        zero_division=0,
    )
)


# STEP 11: DISPLAY THE CONFUSION MATRIX

cm = confusion_matrix(y_test, y_pred, labels=[0, 1])

print("\nConfusion Matrix:")
print(cm)

cm_display = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=["Safe", "Phishing"],
)

cm_display.plot(cmap="Blues", values_format="d")
plt.title("Phishing Email Detection - Confusion Matrix")
plt.tight_layout()
plt.show()


# STEP 12: TEST A NEW EMAIL ENTERED BY THE USER

print("\n========== TEST YOUR EMAIL ==========")
new_email = input("Enter email content: ").strip()

if not new_email:
    print("No email content entered. Exiting.")
else:
    new_text_vectorized = vectorizer.transform([new_email])
    new_features = csr_matrix([extract_features(new_email)])

    new_email_final = hstack(
        [new_text_vectorized, new_features]
    )

    prediction = model.predict(new_email_final)
    probability = model.predict_proba(new_email_final)[0]

    print("\n========== DETECTION RESULT ==========")

    if prediction[0] == 1:
        print("WARNING: PHISHING EMAIL DETECTED!")
    else:
        print("RESULT: SAFE / LEGITIMATE EMAIL")

    print(f"Phishing probability: {probability[1] * 100:.2f}%")
    print(f"Safe probability: {probability[0] * 100:.2f}%")

    extracted = extract_features(new_email)

    print("\nEmail Feature Analysis:")
    print(f"URLs detected: {extracted[0]}")
    print(f"Suspicious keyword matches: {extracted[1]}")
    print(f"Contains IP address URL: {bool(extracted[2])}")
    print(f"Contains URL shortener: {bool(extracted[3])}")
    print(f"HTTPS links detected: {extracted[4]}")

print("\n========== PROGRAM COMPLETED ==========")
