# Spotify Tuner - Backend

Tuner is a powerful tool that elevates your Spotify experience by curating personalized playlists with a seamless emotional flow. Using unsupervised machine learning, it clusters songs based on their emotional essence. Tuner then constructs a weighted distance matrix, calculating the emotional distances between tracks. This matrix is built using a supervised learning model that continuously adapts, adjusting weights to understand how audio features relate to emotion. By minimizing emotional drag between consecutive tracks, Whether you're a music enthusiast, playlist curator, or someone who appreciates harmonious listening, Tuner unlocks the full potential of your Spotify playlists.

## Run tuner locally

Before running the project create a supabase account, create project a new project then copy neccessary variable and add them to your .env file. For reference on what env variable are required refer .env.example file. Also make sure to use a virtual environment.

Clone the project

```bash
  git clone https://github.com/dassaswat/tuner-backend.git
```

Go to the project directory

```bash
  cd tuner-backend
```

Install dependencies

```bash
  pip3 install -r requirements.txt
```

Run database migrations

```bash
  alembic upgrade head
```

Start the server

```bash
  uvicorn main:app --reload
  # Now either visit http://127.0.0.1:8000/docs for api docs
  # Or start the client and have fun
```

## Tech Stack

**Server:** Python, FastAPI, Supabase

## Authors

- [@saswatdas](https://www.github.com/dassaswat)
