import random

def predict(application):
    # Dummy model for predicting credit score and comments
    credit_score = random.randint(300, 850)
    comments = "Good" if credit_score > 650 else "Needs Improvement"
    return {
        'credit_score': credit_score,
        'comments': comments
    }
