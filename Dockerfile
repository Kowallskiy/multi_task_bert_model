FROM python:3.9-slim

WORKDIR /app

COPY . .

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Make port available to the world outside this container
EXPOSE 5000

CMD ["python", "./src/inference.py"]
