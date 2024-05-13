# Spotify Tuner - Backend

Tuner is a powerful tool that elevates your Spotify experience by curating personalized playlists with a seamless emotional flow. Using unsupervised machine learning, it clusters songs based on their emotional essence. Tuner then constructs a weighted distance matrix, calculating the emotional distances between tracks. This matrix is built using a supervised learning model that continuously adapts, adjusting weights to understand how audio features relate to emotion. This allows it to minimize emotional drag between consecutive tracks. Whether you're a music enthusiast, playlist curator, or someone who appreciates harmonious listening, Tuner unlocks the full potential of your Spotify playlists.

## Run tuner locally

Before running the project, create a Supabase account at https://supabase.com and set up a new project. Copy the necessary environment variables from the Supabase project settings and paste them into a .env file in your project directory. Refer to the .env.example file for the required environment variables. Additionally, ensure you're using a virtual environment for your project.

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
```
