# Use the python:slim-bullseye image
FROM python:slim-bullseye

# Set the working directory in the container
WORKDIR /app

# Copy the requirements.txt and install dependencies
COPY requirements.txt requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application files
COPY . .

# Set the environment variables from the .env file
COPY .env .env

# Run the bot.py script
CMD ["python", "bot.py"]