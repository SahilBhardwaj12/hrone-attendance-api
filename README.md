# HROne Software Engineer Trainee Assignment

## Overview

This project provides an employee and attendance management API built using FastAPI and MongoDB. It includes employee management, attendance punch-in and punch-out, attendance regularization, and analytics endpoints.

## Technology Stack

- Python
- FastAPI
- MongoDB
- PyMongo
- Pydantic
- Uvicorn

## Prerequisites

- Python installed
- MongoDB instance accessible to the application
- Required Python packages listed in `requirements.txt`

## Setup

1. Clone or download the project.
2. Open a terminal in the project directory.
3. Create and activate a virtual environment.
4. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

5. Configure the MongoDB connection using the environment variables expected by the application. Do not commit credentials or `.env` secrets.

6. Start the API:

   ```bash
   uvicorn app.main:app --reload --port 8000
   ```

## API Documentation

When the server is running, open:

- Swagger UI: `http://localhost:8000/docs`
- OpenAPI schema: `http://localhost:8000/openapi.json`

## Main Features

- Employee creation and listing
- Attendance punch-in and punch-out
- Attendance history and regularization
- Employee monthly analytics
- Department attendance summary
- Department attendance trends
- Late-arrival leaderboard
- Administrative query explain endpoints

## Validation

The API validates request parameters, including month formats, and returns appropriate HTTP errors for invalid input and nonexistent employees.

## Project Documentation

- `REVIEW.md`: Code review notes
- `DECISIONS.md`: Implementation decisions
- `requirements.txt`: Python dependencies