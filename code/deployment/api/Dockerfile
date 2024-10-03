# fastapi Dockerfile
FROM python:3.9

# Set the working directory
#WORKDIR /app

# Copy the current directory contents into the container at /app
COPY ../ /app

# Install the necessary packages
RUN pip install fastapi uvicorn scikit-learn joblib

# Expose port 80 to the outside world
EXPOSE 80

# Run the FastAPI server
CMD ["uvicorn", "app.code.deployment.api.main:app", "--host", "0.0.0.0", "--port", "80"]