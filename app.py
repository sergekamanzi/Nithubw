from fastapi import FastAPI
from pydantic import BaseModel
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from fastapi.middleware.cors import CORSMiddleware

# Load the trained model and vectorizer
model = pickle.load(open("sentiment_model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

# Initialize FastAPI app
app = FastAPI()

# Enable CORS for frontend integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Define request model
class SentimentRequest(BaseModel):
    text: str

# Define response model
class SentimentResponse(BaseModel):
    sentiment: str

# Sentiment prediction endpoint
@app.post("/predict", response_model=SentimentResponse)
def predict_sentiment(request: SentimentRequest):
    text_tfidf = vectorizer.transform([request.text])
    sentiment = model.predict(text_tfidf)[0]
    return SentimentResponse(sentiment=sentiment)

# Root endpoint
@app.get("/")
def home():
    return {"message": "Sentiment Analysis API is running!"}
